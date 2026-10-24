import pytest
from core.storage import Store


def test_frontend_and_health(client, monkeypatch):
    from core.llm.client import LocalLLM

    monkeypatch.setattr(LocalLLM, "models", lambda self: ["qwen2.5:7b"])
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["ollama"] == "ready"
    assert "script-src" in client.get("/").headers["content-security-policy"]


def test_upload_validated_safe_path(client, app, wav):
    r = client.post("/api/process", files={"file": ("../../escape.wav", wav)})
    assert r.status_code == 202
    m = r.json()
    assert m["title"] == "escape.wav"
    directory = app.state.store.directory(m["id"])
    assert (directory / "audio.wav").read_bytes() == wav
    assert not (directory.parent / "escape.wav").exists()
    assert app.state.jobs.submitted == [(m["id"], "process")]
    assert client.get("/api/meetings").json()[0]["id"] == m["id"]


@pytest.mark.parametrize(
    "filename,content,status",
    [
        ("file.exe", b"x", 415),
        ("broken.wav", b"bad data", 422),
        ("empty.mp3", b"", 422),
        ("large.wav", b"x" * 50001, 413),
    ],
)
def test_upload_rejections(client, app, filename, content, status):
    r = client.post("/api/process", files={"file": (filename, content)})
    assert r.status_code == status
    assert app.state.store.list() == []
    assert not app.state.jobs.slot.locked()
    assert not list(app.state.store.root.glob("*/audio.*"))


def test_busy(client, app, wav):
    app.state.jobs.slot.acquire()
    try:
        assert (
            client.post("/api/process", files={"file": ("x.wav", wav)}).status_code
            == 409
        )
    finally:
        app.state.jobs.slot.release()


def test_cross_origin(client, wav):
    assert (
        client.post(
            "/api/process",
            headers={"Origin": "https://evil.example"},
            files={"file": ("x.wav", wav)},
        ).status_code
        == 403
    )
    assert (
        client.get("/api/meetings", headers={"Host": "evil.example"}).status_code == 400
    )


def test_stream_limit(client):
    def body():
        for _ in range(120):
            yield b"x" * 1024

    r = client.post(
        "/api/process",
        content=body(),
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 413


def test_persistence_and_safe_delete(client, app, meeting, tmp_path):
    store = app.state.store
    store.save(meeting)
    directory = store.directory(meeting["id"])
    directory.mkdir()
    (directory / "audio.wav").write_bytes(b"owned")
    outside = tmp_path / "personal.wav"
    outside.write_bytes(b"private")
    assert Store(store.root).get(meeting["id"])["report"] == meeting["report"]
    assert client.delete("/api/meetings/not-a-uuid").status_code == 404
    assert client.get("/api/meetings/%2e%2e%2fpersonal.wav").status_code == 404
    assert client.delete("/api/meetings/" + meeting["id"]).status_code == 204
    assert not directory.exists()
    assert outside.read_bytes() == b"private"
    assert client.get("/api/meetings/" + meeting["id"]).status_code == 404


def test_recovery(app, meeting):
    meeting["status"] = "processing"
    app.state.store.save(meeting)
    app.state.store.recover()
    assert app.state.store.get(meeting["id"])["error"] == "interrupted"


def test_rename_preserves_original(client, app, meeting):
    app.state.store.save(meeting)
    r = client.patch(
        "/api/meetings/" + meeting["id"] + "/speakers",
        json={"names": {"Speaker 1": "Анна"}},
    )
    assert r.status_code == 200
    assert r.json()["raw_data"][0]["speaker"] == "Speaker 1"
    assert "Анна" in client.get("/api/meetings/" + meeting["id"] + "/export/txt").text
    assert (
        client.patch(
            "/api/meetings/" + meeting["id"] + "/speakers",
            json={"names": {"invented": "X"}},
        ).status_code
        == 422
    )


def test_retry_analysis(client, app, meeting):
    meeting["status"] = "partial"
    app.state.store.save(meeting)
    assert client.post("/api/meetings/" + meeting["id"] + "/retry").status_code == 202
    assert app.state.jobs.submitted[-1][1] == "analyze"


def test_exports_api(client, app, meeting):
    app.state.store.save(meeting)
    for kind in ("txt", "md", "json", "csv", "pdf"):
        response = client.get("/api/meetings/" + meeting["id"] + "/export/" + kind)
        assert response.status_code == 200
        assert len(response.content) > 30
        assert "attachment" in response.headers["content-disposition"]
    assert (
        client.get("/api/meetings/" + meeting["id"] + "/export/exe").status_code == 422
    )


def test_unrelated_chat_abstains(client, app, meeting):
    app.state.store.save(meeting)
    result = client.post(
        "/api/meetings/" + meeting["id"] + "/chat",
        json={"question": "astronomy galaxies"},
    )
    assert result.status_code == 200
    assert result.json()["insufficient_evidence"]
    assert result.json()["sources"] == []


def test_delete_symlink_rejected(app, meeting, tmp_path):
    app.state.store.save(meeting)
    directory = app.state.store.directory(meeting["id"])
    directory.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        app.state.store.delete(meeting["id"])
