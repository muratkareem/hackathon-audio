"""Single-user loopback API. Run exactly one Uvicorn worker."""

import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from core.audio import AudioError, EXTENSIONS, validate_audio
from core.config import ROOT, Settings
from core.exports import export
from core.jobs import Jobs
from core.limits import BodyLimitMiddleware
from core.llm.chat import MeetingRAG
from core.llm.client import InferenceError, LocalLLM
from core.schemas import ChatRequest, RenameRequest
from core.storage import Store


def create_app(settings=None, jobs_factory=Jobs):
    settings = settings or Settings()
    store = Store(settings.data_dir)
    jobs = jobs_factory(store, settings)

    @asynccontextmanager
    async def lifespan(app):
        store.recover()
        yield
        jobs.close()

    app = FastAPI(title="Steppe Hub", lifespan=lifespan)
    app.state.store, app.state.jobs = store, jobs
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_bytes + 65536)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def protect(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "origin_not_allowed"}, status_code=403)
        length = request.headers.get("content-length")
        if length and (
            not length.isdigit() or int(length) > settings.max_upload_bytes + 65536
        ):
            return JSONResponse({"detail": "upload_too_large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    def get(meeting_id):
        try:
            meeting = store.get(meeting_id)
        except ValueError:
            raise HTTPException(404, "meeting_not_found")
        if meeting is None:
            raise HTTPException(404, "meeting_not_found")
        return meeting

    def acquire():
        if not jobs.slot.acquire(blocking=False):
            raise HTTPException(409, "busy")

    @app.get("/api/health")
    def health():
        try:
            models = LocalLLM(settings).models()
            ollama = "ready" if settings.llm_model in models else "model_missing"
        except InferenceError:
            models, ollama = [], "unavailable"
        return {
            "status": "ok",
            "ollama": ollama,
            "models": models,
            "llm_model": settings.llm_model,
            "whisper_model": settings.whisper_model,
            "device": "cpu",
            "compute_type": settings.compute_type,
            "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
            "retain_audio": settings.retain_audio,
            "max_upload_bytes": settings.max_upload_bytes,
            "diarization": settings.diarization,
        }

    @app.get("/api/meetings")
    def meetings():
        return [
            {
                k: m.get(k)
                for k in (
                    "id",
                    "title",
                    "status",
                    "stage",
                    "progress",
                    "created",
                    "error",
                )
            }
            for m in store.list()
        ]

    @app.get("/api/meetings/{meeting_id}")
    def meeting(meeting_id: str):
        return get(meeting_id)

    @app.post("/api/process", status_code=202)
    async def upload(
        file: UploadFile = File(...),
        audio_language: str = Form("auto"),
        summary_language: str = Form("en"),
    ):
        if audio_language not in ("auto", "en", "ru") or summary_language not in (
            "en",
            "ru",
        ):
            raise HTTPException(422, "invalid_language")
        extension = Path(file.filename or "").suffix.lower()
        if extension not in EXTENSIONS:
            raise HTTPException(415, "unsupported_format")
        acquire()
        meeting_id = str(uuid4())
        directory = store.directory(meeting_id)
        submitted = False
        try:
            directory.mkdir(mode=0o700)
            path = directory / ("audio" + extension)
            size = 0
            with path.open("xb") as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, "upload_too_large")
                    target.write(chunk)
            duration = await asyncio.to_thread(
                validate_audio, path, settings.max_duration
            )
            meeting = {
                "id": meeting_id,
                "title": Path((file.filename or "Meeting").replace("\\", "/")).name[
                    :150
                ],
                "extension": extension,
                "duration": duration,
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "audio_language": audio_language,
                "summary_language": summary_language,
                "raw_data": [],
                "speaker_names": {},
                "warnings": [],
                "report": None,
                "error": None,
            }
            store.save(meeting)
            jobs.submit(meeting)
            submitted = True
            return meeting
        except AudioError as exc:
            raise HTTPException(422, str(exc))
        finally:
            await file.close()
            if not submitted:
                shutil.rmtree(directory, ignore_errors=True)
                jobs.slot.release()

    @app.post("/api/meetings/{meeting_id}/retry", status_code=202)
    def retry(meeting_id: str):
        meeting = get(meeting_id)
        if meeting["status"] not in ("failed", "partial"):
            raise HTTPException(409, "retry_unavailable")
        mode = "analyze" if meeting["raw_data"] else "process"
        if (
            mode == "process"
            and not (
                store.directory(meeting_id) / ("audio" + meeting["extension"])
            ).exists()
        ):
            raise HTTPException(409, "reupload_required")
        acquire()
        meeting.update(status="queued", error=None)
        store.save(meeting)
        jobs.submit(meeting, mode)
        return meeting

    @app.post("/api/meetings/{meeting_id}/chat")
    def chat(meeting_id: str, req: ChatRequest):
        meeting = get(meeting_id)
        if not meeting["raw_data"]:
            raise HTTPException(409, "empty_transcript")
        acquire()
        try:
            rag = MeetingRAG()
            # Use aliases for retrieval without overwriting original diarization labels.
            rag.load(
                {
                    **meeting,
                    "raw_data": [
                        {
                            **s,
                            "speaker": meeting["speaker_names"].get(
                                s["speaker"], s["speaker"]
                            ),
                        }
                        for s in meeting["raw_data"]
                    ],
                }
            )
            return rag.ask(req.question, settings, req.language)
        except InferenceError as exc:
            raise HTTPException(503, str(exc))
        finally:
            jobs.slot.release()

    @app.patch("/api/meetings/{meeting_id}/speakers")
    def rename(meeting_id: str, req: RenameRequest):
        meeting = get(meeting_id)
        if meeting["status"] in ("queued", "processing"):
            raise HTTPException(409, "busy")
        labels = {s["speaker"] for s in meeting["raw_data"] if s["speaker"]}
        if not set(req.names).issubset(labels):
            raise HTTPException(422, "invalid_speaker")
        meeting["speaker_names"] = req.names
        store.save(meeting)
        return meeting

    @app.delete("/api/meetings/{meeting_id}", status_code=204)
    def delete(meeting_id: str):
        meeting = get(meeting_id)
        if meeting["status"] in ("queued", "processing"):
            raise HTTPException(409, "busy")
        store.delete(meeting_id)
        return Response(status_code=204)

    @app.get("/api/meetings/{meeting_id}/audio")
    def audio(meeting_id: str):
        meeting = get(meeting_id)
        path = store.directory(meeting_id) / ("audio" + meeting["extension"])
        if not settings.retain_audio or not path.is_file():
            raise HTTPException(404, "audio_not_retained")
        return FileResponse(path)

    @app.get("/api/meetings/{meeting_id}/export/{kind}")
    def download(meeting_id: str, kind: str):
        meeting = get(meeting_id)
        try:
            content, media = export(meeting, kind, settings)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return Response(
            content,
            media_type=media,
            headers={
                "Content-Disposition": f'attachment; filename="meeting-{meeting_id}.{kind}"'
            },
        )

    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


app = create_app()
