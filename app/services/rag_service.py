from pathlib import Path
import re

from app.services.embedding_service import generate_embeddings
from app.services.vector_store import VectorStore
from app.services.llm_service import generate_answer
from app.services.pdf_parser import extract_pdf_links


NOT_AVAILABLE_MESSAGE = (
    "The answer is not available in the uploaded documents."
)


def get_document_links(filename):
    pdf_path = Path("data/uploads") / filename

    if not pdf_path.exists():
        return []

    if pdf_path.suffix.lower() != ".pdf":
        return []

    return extract_pdf_links(pdf_path)


def get_all_document_chunks(vector_store, filename):
    result = vector_store.collection.get(
        where={"filename": filename},
        include=["documents", "metadatas"]
    )

    return (
        result.get("documents", []),
        result.get("metadatas", [])
    )


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_summary_question(question):
    text = normalize_text(question)

    terms = [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "document about",
        "paper about",
        "main points",
        "key points",
        "main idea",
        "overall idea",
        "overall summary",
        "describe the document",
        "describe the paper",
        "explain the document",
        "explain the paper"
    ]

    return any(term in text for term in terms)


def is_comparison_question(question):
    text = normalize_text(question)

    terms = [
        "difference",
        "differences",
        "compare",
        "comparison",
        "compared",
        "versus",
        "vs",
        "distinguish",
        "distinction"
    ]

    return any(term in text for term in terms)


def is_list_question(question):
    text = normalize_text(question)

    terms = [
        "all models",
        "models used",
        "what models",
        "which models",
        "architectures",
        "all methods",
        "methods used",
        "what methods",
        "which methods",
        "all techniques",
        "techniques used",
        "what techniques",
        "which techniques",
        "list models",
        "list methods",
        "what datasets",
        "datasets used",
        "all datasets",
        "what metrics",
        "metrics used"
    ]

    return any(term in text for term in terms)


def is_short_question(question):
    words = normalize_text(question).split()

    return len(words) <= 4


def get_llm_limit(question):
    if is_summary_question(question):
        return 6

    if is_list_question(question):
        return 6

    if is_comparison_question(question):
        return 5

    return 3


def get_candidate_limit(question):
    if is_summary_question(question):
        return 10

    if is_list_question(question):
        return 10

    if is_comparison_question(question):
        return 8

    return 6


def keyword_score(question, document):
    question_text = normalize_text(question)
    document_text = normalize_text(document)

    words = [
        word
        for word in question_text.split()
        if len(word) > 1
    ]

    score = 0

    for word in words:
        if word in document_text:
            score += 1

    return score


def phrase_score(question, document):
    question_text = normalize_text(question)
    document_text = normalize_text(document)

    words = question_text.split()

    if len(words) < 2:
        return 0

    score = 0

    for size in (4, 3, 2):
        if len(words) < size:
            continue

        for i in range(len(words) - size + 1):
            phrase = " ".join(
                words[i:i + size]
            )

            if (
                len(phrase) > 3
                and phrase in document_text
            ):
                score += size

    return score


def question_type_score(question, document):
    question_text = normalize_text(question)
    document_text = normalize_text(document)

    score = 0

    groups = {
        "title": [
            "title",
            "name",
            "heading"
        ],
        "author": [
            "author",
            "authors",
            "written by"
        ],
        "abstract": [
            "abstract"
        ],
        "model": [
            "model",
            "models",
            "architecture",
            "architectures"
        ],
        "dataset": [
            "dataset",
            "datasets",
            "data"
        ],
        "result": [
            "result",
            "results",
            "accuracy",
            "performance",
            "precision",
            "recall",
            "f1"
        ],
        "method": [
            "method",
            "methods",
            "methodology",
            "approach"
        ],
        "objective": [
            "objective",
            "objectives",
            "purpose",
            "goal"
        ]
    }

    for key, terms in groups.items():

        if any(
            term in question_text
            for term in terms
        ):
            for term in terms:
                if term in document_text:
                    score += 1

    return score


def page_priority(metadata):
    page = metadata.get("page")

    try:
        page = int(page)

        if page == 1:
            return 0.5

        if page == 2:
            return 0.2

    except (
        ValueError,
        TypeError
    ):
        pass

    return 0


def get_document_metadata_answer(
    question,
    filename,
    metadatas
):
    if not metadatas:
        return None

    question_text = normalize_text(question)

    file_type = str(
        metadatas[0].get(
            "file_type",
            "unknown"
        )
    ).lower()

    if any(
        term in question_text
        for term in [
            "page",
            "pages",
            "page count",
            "number of pages",
            "how many pages"
        ]
    ):
        if file_type == "pdf":

            pages = set()

            for metadata in metadatas:

                page = metadata.get("page")

                try:
                    if page is not None:
                        pages.add(int(page))
                except (
                    ValueError,
                    TypeError
                ):
                    pass

            if pages:

                return {
                    "answer": (
                        f"The document contains "
                        f"{max(pages)} pages."
                    ),
                    "source": {
                        "filename": filename,
                        "page": "Document metadata",
                        "section": "Document Information",
                        "file_type": file_type,
                        "chunk_id": "metadata",
                        "distance": None
                    }
                }

    if any(
        term in question_text
        for term in [
            "file name",
            "filename",
            "document name"
        ]
    ):
        return {
            "answer": (
                f"The document name is {filename}."
            ),
            "source": {
                "filename": filename,
                "page": "Document metadata",
                "section": "Document Information",
                "file_type": file_type,
                "chunk_id": "metadata",
                "distance": None
            }
        }

    if any(
        term in question_text
        for term in [
            "file type",
            "document type",
            "what type of file"
        ]
    ):
        return {
            "answer": (
                f"The document is a "
                f"{file_type.upper()} file."
            ),
            "source": {
                "filename": filename,
                "page": "Document metadata",
                "section": "Document Information",
                "file_type": file_type,
                "chunk_id": "metadata",
                "distance": None
            }
        }

    if any(
        term in question_text
        for term in [
            "sections",
            "what sections",
            "which sections"
        ]
    ):
        sections = []

        for metadata in metadatas:

            section = metadata.get("section")

            if section:

                section = str(section)

                if section not in sections:
                    sections.append(section)

        if sections:

            return {
                "answer": (
                    "The document contains "
                    "the following sections:\n\n"
                    + "\n".join(
                        f"- {section}"
                        for section in sections
                    )
                ),
                "source": {
                    "filename": filename,
                    "page": "Document metadata",
                    "section": "Document Structure",
                    "file_type": file_type,
                    "chunk_id": "metadata",
                    "distance": None
                }
            }

    if any(
        term in question_text
        for term in [
            "sheets",
            "what sheets",
            "which sheets"
        ]
    ):
        sheets = []

        for metadata in metadatas:

            sheet = metadata.get("sheet")

            if sheet:

                sheet = str(sheet)

                if sheet not in sheets:
                    sheets.append(sheet)

        if sheets:

            return {
                "answer": (
                    f"The document contains "
                    f"{len(sheets)} sheets:\n\n"
                    + "\n".join(
                        f"- {sheet}"
                        for sheet in sheets
                    )
                ),
                "source": {
                    "filename": filename,
                    "page": "Document metadata",
                    "section": "Document Structure",
                    "file_type": file_type,
                    "chunk_id": "metadata",
                    "distance": None
                }
            }

    return None


def retrieve_relevant_chunks(
    question,
    filename,
    vector_store
):
    question_text = normalize_text(question)

    front_terms = [
        "title",
        "author",
        "authors",
        "abstract",
        "publication",
        "published",
        "year",
        "affiliation"
    ]

    is_front_question = any(
        term in question_text
        for term in front_terms
    )

    llm_limit = get_llm_limit(question)

    if is_front_question:

        result = vector_store.collection.get(
            where={
                "filename": filename
            },
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = result.get(
            "documents",
            []
        )

        metadatas = result.get(
            "metadatas",
            []
        )

        candidates = []

        for index, (document, metadata) in enumerate(
            zip(documents, metadatas)
        ):

            page = metadata.get("page")

            try:
                page = int(page)
            except (ValueError, TypeError):
                continue

            if page not in (1, 2):
                continue

            text = normalize_text(document)

            score = 0

            if page == 1:
                score += 100

            score += max(
                0,
                20 - index
            )

            if "abstract" in question_text:
                if "abstract" in text:
                    score += 50

            if any(
                word in question_text
                for word in [
                    "author",
                    "authors"
                ]
            ):
                if any(
                    word in text
                    for word in [
                        "author",
                        "authors",
                        "university",
                        "institute",
                        "department",
                        "email"
                    ]
                ):
                    score += 50

            if "title" in question_text:
                score += max(
                    0,
                    15 - index
                )

            candidates.append({
                "document": document,
                "metadata": metadata,
                "distance": None,
                "score": score
            })

        candidates.sort(
            key=lambda x: -x["score"]
        )

        return candidates[:llm_limit]

    candidate_limit = get_candidate_limit(
        question
    )

    query_embedding = generate_embeddings(
        [question]
    )[0]

    results = vector_store.search(
        query_embedding,
        n_results=candidate_limit,
        filename=filename
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    candidates = []

    for index, document in enumerate(
        documents
    ):

        if index >= len(metadatas):
            continue

        metadata = metadatas[index]

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        semantic_score = 0

        if distance is not None:
            semantic_score = (
                1 /
                (1 + float(distance))
            )

        lexical_score = keyword_score(
            question,
            document
        )

        phrase = phrase_score(
            question,
            document
        )

        type_score = question_type_score(
            question,
            document
        )

        priority = page_priority(
            metadata
        )

        score = (
            semantic_score
            + lexical_score * 0.12
            + phrase * 0.15
            + type_score * 0.15
            + priority
        )

        candidates.append({
            "document": document,
            "metadata": metadata,
            "distance": distance,
            "score": score
        })

    candidates.sort(
        key=lambda x: -x["score"]
    )

    selected = []
    used_ids = set()
    used_pages = set()

    if (
        is_summary_question(question)
        or is_list_question(question)
    ):

        for candidate in candidates:

            metadata = candidate["metadata"]

            chunk_id = metadata.get(
                "chunk_id"
            )

            page = metadata.get(
                "page"
            )

            if chunk_id in used_ids:
                continue

            if page in used_pages:
                continue

            selected.append(candidate)

            used_ids.add(chunk_id)
            used_pages.add(page)

            if len(selected) >= llm_limit:
                break

    for candidate in candidates:

        if len(selected) >= llm_limit:
            break

        chunk_id = candidate[
            "metadata"
        ].get("chunk_id")

        if chunk_id in used_ids:
            continue

        selected.append(candidate)
        used_ids.add(chunk_id)

    return selected


def build_context(candidates):
    parts = []

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        metadata = candidate["metadata"]

        header = (
            f"PASSAGE {index}\n"
            f"Page: "
            f"{metadata.get('page', 'Unknown')}\n"
            f"Section: "
            f"{metadata.get('section', 'General')}\n"
        )

        if metadata.get("sheet"):
            header += (
                f"Sheet: "
                f"{metadata['sheet']}\n"
            )

        parts.append(
            header
            + "\n"
            + candidate["document"]
        )

    return "\n\n".join(parts)


def build_prompt(
    question,
    filename,
    context
):
    return f"""
You are ELITE Verity, an evidence-grounded
document intelligence assistant.

Document:
{filename}

Relevant passages:
{context}

Question:
{question}

Answer using only the information supported
by the provided passages.

Give a deep, accurate and complete answer.

For short factual questions, answer directly.

For list questions, include every item supported
by the passages.

For comparison questions, explain the important
differences clearly.

For why/how questions, explain the reasoning or
process when supported.

For summaries, provide a structured and meaningful
summary of the information available in the
provided passages.

Preserve technical terminology, abbreviations,
formulas and numerical values accurately.

Do not invent information.
Do not use outside knowledge.

If the provided passages do not support the answer,
say exactly:

The answer is not available in the uploaded documents.

Do not mention retrieval, embeddings, chunks,
ChromaDB, prompts or internal implementation.

Return only the answer.
"""


def answer_question(
    question,
    filename
):
    question = str(
        question
    ).strip()

    if not question:

        return {
            "question": question,
            "answer": "Please enter a question.",
            "source": None,
            "distance": None
        }

    vector_store = VectorStore()

    all_documents, all_metadatas = (
        get_all_document_chunks(
            vector_store,
            filename
        )
    )

    if not all_documents:

        return {
            "question": question,
            "answer": NOT_AVAILABLE_MESSAGE,
            "source": None,
            "distance": None
        }

    metadata_result = (
        get_document_metadata_answer(
            question,
            filename,
            all_metadatas
        )
    )

    if metadata_result:

        return {
            "question": question,
            "answer": metadata_result["answer"],
            "source": metadata_result["source"],
            "distance": None
        }

    candidates = retrieve_relevant_chunks(
        question,
        filename,
        vector_store
    )

    if not candidates:

        return {
            "question": question,
            "answer": NOT_AVAILABLE_MESSAGE,
            "source": None,
            "distance": None
        }

    context = build_context(
        candidates
    )

    prompt = build_prompt(
        question,
        filename,
        context
    )

    try:

        answer = generate_answer(
            prompt
        )

        answer = str(
            answer
        ).strip()

    except Exception as error:

        error_text = str(
            error
        ).lower()

        if (
            "429" in error_text
            or "rate_limit" in error_text
            or "rate limit" in error_text
            or "tokens per day" in error_text
        ):

            return {
                "question": question,
                "answer": (
                    "The AI generation service has "
                    "temporarily reached its token limit. "
                    "Please try again after the limit resets."
                ),
                "source": None,
                "distance": None
            }

        raise

    if not answer:

        answer = NOT_AVAILABLE_MESSAGE

    if (
        answer.lower()
        == NOT_AVAILABLE_MESSAGE.lower()
    ):

        return {
            "question": question,
            "answer": NOT_AVAILABLE_MESSAGE,
            "source": None,
            "distance": None
        }

    best = candidates[0]

    metadata = best["metadata"]

    distance = best.get(
        "distance"
    )

    source = {
        "filename": metadata.get(
            "filename",
            filename
        ),
        "page": metadata.get(
            "page",
            "Unknown"
        ),
        "section": metadata.get(
            "section",
            "General"
        ),
        "file_type": metadata.get(
            "file_type",
            "unknown"
        ),
        "chunk_id": metadata.get(
            "chunk_id",
            "Unknown"
        ),
        "distance": distance
    }

    if metadata.get("sheet"):
        source["sheet"] = metadata["sheet"]

    return {
        "question": question,
        "answer": answer,
        "source": source,
        "distance": distance
    }