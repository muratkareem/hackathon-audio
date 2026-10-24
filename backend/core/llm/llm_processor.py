import json
from core.llm.client import LocalLLM, InferenceError
from core.llm.prompts import SUMMARY
from core.schemas import Report


def chunks(segments, budget=10000):
    current, size = [], 0
    for segment in segments:
        cost = len(json.dumps(segment, ensure_ascii=False))
        if current and size + cost > budget:
            yield current
            current, size = [], 0
        current.append(segment)
        size += cost
    if current:
        yield current


def check_references(report, allowed):
    for field in ("discussion", "decisions", "actions", "open_questions"):
        for item in getattr(report, field):
            if not set(item.references).issubset(allowed):
                raise InferenceError("invalid_references")
    return report


class ProtocolGenerator:
    def __init__(self, settings):
        self.client = LocalLLM(settings)

    def generate_protocol(self, segments, language="en"):
        reports = []
        for group in chunks(segments):
            report = self.client.generate(
                Report,
                SUMMARY,
                json.dumps(
                    {"language": language, "segments": group}, ensure_ascii=False
                ),
            )
            reports.append(check_references(report, {s["id"] for s in group}))
        if not reports:
            raise InferenceError("empty_transcript")
        # Bounded tree synthesis: no giant final prompt and no silently dropped chunks.
        while len(reports) > 1:
            merged = []
            for i in range(0, len(reports), 2):
                pair = reports[i : i + 2]
                if len(pair) == 1:
                    merged.append(pair[0])
                    continue
                allowed = {
                    ref
                    for r in pair
                    for field in (
                        "discussion",
                        "decisions",
                        "actions",
                        "open_questions",
                    )
                    for item in getattr(r, field)
                    for ref in item.references
                }
                content = json.dumps(
                    {
                        "language": language,
                        "partial_reports": [r.model_dump() for r in pair],
                    },
                    ensure_ascii=False,
                )
                if len(content) > 26000:
                    raise InferenceError("summary_context_limit")
                merged.append(
                    check_references(
                        self.client.generate(
                            Report,
                            SUMMARY + " Synthesize these partial reports.",
                            content,
                        ),
                        allowed,
                    )
                )
            reports = merged
        return reports[0].model_dump()
