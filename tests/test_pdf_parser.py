from app.services.pdf_parser import extract_pdf_content


pdf_path = "sample.pdf"

pages = extract_pdf_content(pdf_path)

for page in pages:
    print(f"\n--- Page {page['page']} ---")
    print(page["text"][:500])

    if page["links"]:
        print("Links:")
        for link in page["links"]:
            print(link)