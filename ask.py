from pathlib import Path

from app.services.document_processor import process_pdf
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import VectorStore
from app.services.rag_service import (
    answer_question,
    get_document_links
)


UPLOAD_DIR = Path("data/uploads")


LINK_KEYWORDS = [
    "link",
    "links",
    "hyperlink",
    "hyperlinks",
    "url",
    "urls",
    "website",
    "websites"
]


def is_link_question(question):

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in LINK_KEYWORDS
    )


def main():

    pdf_files = list(
        UPLOAD_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print(
            "No PDF files found in data/uploads."
        )

        return

    print()
    print("=" * 70)
    print("LOCAL PDF AI ASSISTANT")
    print("=" * 70)

    print(
        "\nAvailable PDF documents:\n"
    )

    for index, pdf in enumerate(
        pdf_files,
        start=1
    ):

        print(
            f"[{index}] {pdf.name}"
        )

    while True:

        try:

            choice = int(
                input(
                    "\nSelect document number: "
                )
            )

            if 1 <= choice <= len(pdf_files):
                break

            print(
                "Invalid selection."
            )

        except ValueError:

            print(
                "Please enter a valid number."
            )

    selected_pdf = pdf_files[
        choice - 1
    ]

    print()
    print("=" * 70)
    print(
        f"SELECTED DOCUMENT: "
        f"{selected_pdf.name}"
    )
    print("=" * 70)

    print(
        "\nProcessing document..."
    )

    try:

        chunks = process_pdf(
            selected_pdf
        )

        if not chunks:

            print()
            print(
                "The PDF contains no extractable text."
            )

            print(
                "Please select a text-based PDF."
            )

            return

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = generate_embeddings(
            texts
        )

        store = VectorStore()

        store.add_chunks(
            chunks,
            embeddings
        )

        print(
            f"Processed chunks: "
            f"{len(chunks)}"
        )

        print(
            "Document is ready."
        )

    except Exception as e:

        print(
            f"\nError processing document: {e}"
        )

        return

    print()
    print("=" * 70)
    print("QUESTION MODE")
    print("=" * 70)

    print(
        f"Document: {selected_pdf.name}"
    )

    print()
    print(
        "Ask as many questions as you want."
    )

    print(
        "Type 'exit' to finish."
    )

    print("=" * 70)

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":

            print(
                "\nExiting..."
            )

            break

        if not question:

            print(
                "Question cannot be empty."
            )

            continue

        if is_link_question(
            question
        ):

            print(
                "\nExtracting hyperlinks..."
            )

            try:

                links = get_document_links(
                    selected_pdf.name
                )

                print()
                print("-" * 70)
                print("HYPERLINKS / URLs")
                print("-" * 70)

                if not links:

                    print(
                        "No hyperlinks or URLs "
                        "found in the document."
                    )

                else:

                    for index, link in enumerate(
                        links,
                        start=1
                    ):

                        print(
                            f"{index}. "
                            f"Page {link['page']} | "
                            f"{link['url']}"
                        )

                print(
                    "-" * 70
                )

            except Exception as e:

                print(
                    f"\nError extracting links: {e}"
                )

            continue

        print(
            "\nGenerating answer..."
        )

        try:

            result = answer_question(
                question,
                selected_pdf.name
            )

            print()
            print("-" * 70)
            print("ANSWER")
            print("-" * 70)

            print(
                result["answer"]
            )

            print()
            print("SOURCE")
            print("-" * 70)

            if result["source"]:

                source = result["source"]

                print(
                    f"Filename : "
                    f"{source['filename']}"
                )

                print(
                    f"Page     : "
                    f"{source['page']}"
                )

                print(
                    f"Section  : "
                    f"{source.get('section', 'General')}"
                )

                print(
                    f"Chunk ID  : "
                    f"{source['chunk_id']}"
                )

                print(
                    f"Distance : "
                    f"{source['distance']:.4f}"
                )

            else:

                print(
                    "No relevant source found."
                )

            print(
                "-" * 70
            )

        except Exception as e:

            print(
                f"\nError generating answer: {e}"
            )


if __name__ == "__main__":
    main()