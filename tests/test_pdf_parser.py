from pathlib import Path

from app.services.pdf_parser import extract_pdf_content, extract_pdf_links


PDF_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "uploads" / "sample.pdf"


def test_extract_pdf_content_returns_pages_and_links():
    pages = extract_pdf_content(PDF_FIXTURE)

    assert pages
    assert [page["page"] for page in pages] == list(range(1, len(pages) + 1))
    assert all(set(page) == {"page", "text", "links"} for page in pages)
    assert all(isinstance(page["text"], str) for page in pages)
    assert all(isinstance(page["links"], list) for page in pages)

    content_links = [link for page in pages for link in page["links"]]
    assert content_links == extract_pdf_links(PDF_FIXTURE)
