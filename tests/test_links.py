from pathlib import Path

from app.services.pdf_parser import extract_pdf_links


PDF_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "uploads" / "sample.pdf"


def test_extract_pdf_links_returns_unique_link_metadata():
    links = extract_pdf_links(PDF_FIXTURE)

    assert isinstance(links, list)
    assert len({(link["page"], link["url"]) for link in links}) == len(links)
    assert all(
        set(link) == {"page", "url", "type"}
        and isinstance(link["page"], int)
        and link["url"]
        and link["type"] in {"clickable", "text"}
        for link in links
    )
