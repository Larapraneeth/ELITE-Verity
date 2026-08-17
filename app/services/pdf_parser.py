import re
import pymupdf


def extract_pdf_content(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)

        pages = []

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")

            pages.append({
                "page": page_number,
                "text": text
            })

        doc.close()

        return pages

    except Exception as e:
        raise ValueError(f"Could not process PDF: {e}")


def extract_pdf_links(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)

        links = []

        url_pattern = re.compile(
            r"https?://[^\s<>\]\)]+"
        )

        for page_number, page in enumerate(doc, start=1):

            page_links = page.get_links()

            for link in page_links:
                uri = link.get("uri")

                if uri:
                    links.append({
                        "page": page_number,
                        "url": uri,
                        "type": "clickable"
                    })

            text = page.get_text("text")

            text_urls = url_pattern.findall(text)

            for url in text_urls:
                url = url.rstrip(".,;:")

                links.append({
                    "page": page_number,
                    "url": url,
                    "type": "text"
                })

        doc.close()

        unique_links = []
        seen = set()

        for link in links:
            key = (link["page"], link["url"])

            if key not in seen:
                seen.add(key)
                unique_links.append(link)

        return unique_links

    except Exception as e:
        raise ValueError(f"Could not extract PDF links: {e}")