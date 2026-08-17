from app.services.pdf_parser import extract_pdf_content


def test_file(path):
    print("\n" + "=" * 60)
    print("Testing:", path)
    print("=" * 60)

    try:
        pages = extract_pdf_content(path)

        print("STATUS: VALID")
        print("Pages:", len(pages))

    except Exception as e:
        print("STATUS: REJECTED")
        print("Reason:", e)


test_file("data/uploads/sample.pdf")