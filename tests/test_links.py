from app.services.pdf_parser import extract_pdf_content


pdf_path = "data/uploads/sample.pdf"

pages = extract_pdf_content(pdf_path)

total_links = 0

print("=" * 70)
print("PDF HYPERLINK EXTRACTION")
print("=" * 70)

for page in pages:
    links = page["links"]

    if links:
        print(f"\nPage {page['page']}:")

        for link in links:
            print(f"  {link}")
            total_links += 1

print("\n" + "=" * 70)
print(f"Total hyperlinks found: {total_links}")
print("=" * 70)