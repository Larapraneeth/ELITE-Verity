from pathlib import Path

from app.services.document_processor import process_pdf


PDF_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "uploads" / "sample.pdf"


def test_process_pdf_preserves_chunk_metadata():
    chunks = process_pdf(PDF_FIXTURE)

    assert chunks
    for chunk in chunks:
        assert {"text", "filename", "page", "section", "file_type", "chunk_id"} <= set(chunk)
        assert chunk["text"]
        assert chunk["filename"] == PDF_FIXTURE.name
        assert isinstance(chunk["page"], int)
        assert chunk["file_type"] == "pdf"
        assert chunk["chunk_id"].startswith(f"{PDF_FIXTURE.stem}_page")
