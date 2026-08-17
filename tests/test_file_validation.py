from pathlib import Path

from app.services.pdf_parser import extract_pdf_content


PDF_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "uploads" / "sample.pdf"


def test_pdf_fixture_is_processable():
    assert PDF_FIXTURE.is_file()
    assert extract_pdf_content(PDF_FIXTURE)
