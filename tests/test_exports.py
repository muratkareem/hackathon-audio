import io
import json
import csv
from pypdf import PdfReader
from core.exports import export
from core.config import ROOT


def test_unicode_exports(settings, meeting):
    text, _ = export(meeting, "txt", settings)
    assert "Анна" in text.decode()
    md, _ = export(meeting, "md", settings)
    assert "00:00:00" in md.decode() and "пятница" in md.decode()
    raw, _ = export(meeting, "json", settings)
    assert json.loads(raw)["raw_data"] == meeting["raw_data"]
    pdf, _ = export(meeting, "pdf", settings)
    reader = PdfReader(io.BytesIO(pdf))
    assert "Учебная" in "".join(p.extract_text() for p in reader.pages)
    assert len(reader.pages) >= 1


def test_csv_formula_neutralized(settings, meeting):
    meeting["report"]["actions"][0]["text"] = '=HYPERLINK("bad")'
    payload, _ = export(meeting, "csv", settings)
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    assert rows[1][0].startswith("'=")
    assert rows[1][1] == "Анна"


def test_localization_keys_and_no_remote_scripts():
    data = json.loads((ROOT / "frontend/i18n.json").read_text())
    assert set(data["en"]) == set(data["ru"])
    assert all(data["ru"].values())
    html = (ROOT / "frontend/index.html").read_text()
    assert "https://" not in html
    js = (ROOT / "frontend/app.js").read_text()
    assert "innerHTML" not in js


def test_csv_headers_follow_report_language(settings, meeting):
    payload, _ = export(meeting, "csv", settings)
    assert (
        payload.decode("utf-8-sig")
        .splitlines()[0]
        .startswith("Задача,Ответственный,Срок,")
    )
    meeting["summary_language"] = "en"
    payload, _ = export(meeting, "csv", settings)
    assert (
        payload.decode("utf-8-sig").splitlines()[0].startswith("Task,Owner,Deadline,")
    )
