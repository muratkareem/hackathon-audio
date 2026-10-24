import csv
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape

LABELS = {
    "en": {
        "overview": "Meeting overview",
        "topics": "Topics",
        "discussion": "Discussion",
        "decisions": "Decisions",
        "actions": "Action items",
        "open_questions": "Open questions",
        "takeaways": "Key takeaways",
        "unknown": "Unknown speaker",
        "none": "Not specified",
    },
    "ru": {
        "overview": "Обзор встречи",
        "topics": "Темы",
        "discussion": "Обсуждение",
        "decisions": "Решения",
        "actions": "Задачи",
        "open_questions": "Открытые вопросы",
        "takeaways": "Главные выводы",
        "unknown": "Спикер не определён",
        "none": "Не указано",
    },
}


def stamp(value):
    seconds = int(value)
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def speaker(meeting, segment):
    key = segment.get("speaker")
    return (
        meeting.get("speaker_names", {}).get(key, key)
        or LABELS[meeting["summary_language"]]["unknown"]
    )


def transcript(meeting):
    return "\n".join(
        f"[{stamp(s['start'])} - {stamp(s['end'])}] {speaker(meeting, s)}: {s['text']}"
        for s in meeting.get("raw_data", [])
    )


def markdown(meeting):
    labels = LABELS[meeting["summary_language"]]
    lines = ["# " + meeting["title"], ""]
    report = meeting.get("report")
    if not report:
        return "\n".join(lines) + transcript(meeting)
    by_id = {s["id"]: s for s in meeting["raw_data"]}
    for key in (
        "overview",
        "topics",
        "discussion",
        "decisions",
        "actions",
        "open_questions",
        "takeaways",
    ):
        lines.extend(["## " + labels[key], ""])
        value = report[key]
        if isinstance(value, str):
            lines.append(value)
        else:
            for item in value:
                if isinstance(item, str):
                    lines.append("- " + item)
                else:
                    text = item["text"]
                    if key == "actions":
                        text += (
                            " | "
                            + (item["owner"] or labels["none"])
                            + " | "
                            + (item["deadline"] or labels["none"])
                        )
                    refs = ", ".join(
                        stamp(by_id[r]["start"])
                        for r in item["references"]
                        if r in by_id
                    )
                    lines.append("- " + text + " [" + refs + "]")
        lines.append("")
    return "\n".join(lines)


def safe_cell(value):
    text = "" if value is None else str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def export(meeting, kind, settings):
    if kind == "txt":
        return transcript(meeting).encode("utf-8"), "text/plain; charset=utf-8"
    if kind == "md":
        return markdown(meeting).encode("utf-8"), "text/markdown; charset=utf-8"
    if kind == "json":
        return json.dumps(meeting, ensure_ascii=False, indent=2).encode(
            "utf-8"
        ), "application/json"
    if kind == "csv":
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(["task", "owner", "deadline", "segment_references"])
        for item in (meeting.get("report") or {}).get("actions", []):
            writer.writerow(
                [safe_cell(item.get(k)) for k in ("text", "owner", "deadline")]
                + [",".join(map(str, item["references"]))]
            )
        return buffer.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8"
    if kind == "pdf":
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

        candidates = [
            settings.pdf_font,
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        font = next((p for p in candidates if p and Path(p).is_file()), None)
        if not font:
            raise ValueError("pdf_font_missing")
        if "MeetingUnicode" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("MeetingUnicode", font))
        buffer = io.BytesIO()
        body = ParagraphStyle(
            "body", fontName="MeetingUnicode", fontSize=10, leading=15, spaceAfter=6
        )
        heading = ParagraphStyle(
            "heading", parent=body, fontSize=14, leading=19, spaceBefore=12
        )
        flow = []
        for line in markdown(meeting).splitlines():
            flow.append(
                Paragraph(
                    escape(line.lstrip("# ")) if line.startswith("#") else escape(line),
                    heading if line.startswith("#") else body,
                )
                if line
                else Spacer(1, 5)
            )
        SimpleDocTemplate(
            buffer,
            title=meeting["title"],
            leftMargin=42,
            rightMargin=42,
            topMargin=42,
            bottomMargin=42,
        ).build(flow)
        return buffer.getvalue(), "application/pdf"
    raise ValueError("unsupported_export")
