import streamlit as st
from pathlib import Path

from app.services.document_processor import process_document
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import VectorStore
from app.services.rag_service import (
    answer_question,
    get_document_links,
    NOT_AVAILABLE_MESSAGE
)


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="ELITE Verity",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded"
)


if "selected_document" not in st.session_state:
    st.session_state.selected_document = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "conversations" not in st.session_state:
    st.session_state.conversations = {}


st.sidebar.title("ELITE Verity")

st.sidebar.caption(
    "Evidence-Grounded Document Intelligence"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Document Workspace"
)


uploaded_file = st.sidebar.file_uploader(
    "Upload a document",
    type=[
        "pdf",
        "txt",
        "csv",
        "xlsx",
        "xlsm",
        "xls"
    ]
)


if uploaded_file:

    save_path = (
        UPLOAD_DIR /
        uploaded_file.name
    )

    if not save_path.exists():

        with open(
            save_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

    st.session_state.selected_document = (
        uploaded_file.name
    )

    if uploaded_file.name not in st.session_state.conversations:

        st.session_state.conversations[
            uploaded_file.name
        ] = []


supported_files = []

for pattern in [
    "*.pdf",
    "*.txt",
    "*.csv",
    "*.xlsx",
    "*.xlsm",
    "*.xls"
]:

    supported_files.extend(
        UPLOAD_DIR.glob(pattern)
    )


supported_files = sorted(
    supported_files,
    key=lambda file: file.name.lower()
)


if not supported_files:

    st.title("ELITE Verity")

    st.caption(
        "Evidence-Grounded Document Intelligence"
    )

    st.info(
        "Upload a document from the sidebar to get started."
    )

    st.stop()


document_names = [
    file.name
    for file in supported_files
]


if (
    st.session_state.selected_document
    not in document_names
):

    st.session_state.selected_document = (
        document_names[0]
    )


selected_document = st.sidebar.selectbox(
    "Select document",
    document_names,
    index=document_names.index(
        st.session_state.selected_document
    )
)


if selected_document not in st.session_state.conversations:

    st.session_state.conversations[
        selected_document
    ] = []


st.session_state.selected_document = (
    selected_document
)


selected_path = (
    UPLOAD_DIR /
    selected_document
)


file_type = (
    selected_path.suffix
    .replace(".", "")
    .upper()
)


st.sidebar.caption(
    f"Selected: {selected_document}"
)

st.sidebar.caption(
    f"Type: {file_type}"
)


process_button = st.sidebar.button(
    "Process Document",
    use_container_width=True
)


if process_button:

    try:

        with st.spinner(
            "Processing document..."
        ):

            chunks = process_document(
                selected_path
            )

            if not chunks:

                st.error(
                    "No extractable content was found."
                )

                st.stop()

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = generate_embeddings(
                texts
            )

            vector_store = VectorStore()

            vector_store.add_chunks(
                chunks,
                embeddings
            )

            st.session_state.processed_files.add(
                selected_document
            )

        st.sidebar.success(
            f"Processed {len(chunks)} chunks"
        )

    except Exception as error:

        st.sidebar.error(
            f"Processing failed: {error}"
        )


is_processed = (
    selected_document
    in st.session_state.processed_files
)


if is_processed:

    st.sidebar.success(
        "Document is ready"
    )

else:

    st.sidebar.warning(
        "Process the document before asking questions."
    )


st.title(
    "ELITE Verity"
)

st.caption(
    "Evidence-Grounded Document Intelligence"
)


if not is_processed:

    st.info(
        "Upload and select a document, then click "
        "'Process Document' in the sidebar."
    )

    st.stop()


st.success(
    f"Ready: {selected_document}"
)

st.divider()


messages = st.session_state.conversations[
    selected_document
]


for message in messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("source")
        ):

            source = message["source"]

            with st.expander(
                "Source Reference"
            ):

                st.write(
                    f"File: "
                    f"{source.get(
                        'filename',
                        'Unknown'
                    )}"
                )

                st.write(
                    f"Page: "
                    f"{source.get(
                        'page',
                        'Unknown'
                    )}"
                )

                st.write(
                    f"Section: "
                    f"{source.get(
                        'section',
                        'General'
                    )}"
                )

                st.write(
                    f"File type: "
                    f"{source.get(
                        'file_type',
                        'Unknown'
                    )}"
                )

                if source.get("sheet"):

                    st.write(
                        f"Sheet: "
                        f"{source['sheet']}"
                    )

                st.write(
                    f"Chunk: "
                    f"{source.get(
                        'chunk_id',
                        'Unknown'
                    )}"
                )

                if (
                    source.get("distance")
                    is not None
                ):

                    st.write(
                        f"Distance: "
                        f"{source['distance']:.4f}"
                    )


question = st.chat_input(
    "Ask anything about your document..."
)


if question:

    user_message = {
        "role": "user",
        "content": question
    }

    st.session_state.conversations[
        selected_document
    ].append(
        user_message
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    link_keywords = [
        "link",
        "links",
        "hyperlink",
        "hyperlinks",
        "url",
        "urls",
        "website",
        "websites"
    ]


    is_link_question = any(
        keyword in question.lower()
        for keyword in link_keywords
    )


    with st.chat_message(
        "assistant"
    ):

        if is_link_question:

            if (
                selected_path.suffix.lower()
                != ".pdf"
            ):

                answer = (
                    "Hyperlink extraction is "
                    "currently available for "
                    "PDF documents."
                )

                st.markdown(
                    answer
                )

            else:

                with st.spinner(
                    "Extracting hyperlinks..."
                ):

                    links = get_document_links(
                        selected_document
                    )

                if not links:

                    answer = (
                        "No hyperlinks or URLs "
                        "were found in this document."
                    )

                    st.markdown(
                        answer
                    )

                else:

                    answer = (
                        f"Found {len(links)} "
                        f"hyperlinks in the document."
                    )

                    st.markdown(
                        f"Found {len(links)} "
                        f"hyperlinks:"
                    )

                    for index, link in enumerate(
                        links,
                        start=1
                    ):

                        url = link["url"]

                        page = link["page"]

                        st.markdown(
                            f"{index}. "
                            f"Page {page}: "
                            f"[{url}]({url})"
                        )

                        answer += (
                            f"\n\n"
                            f"{index}. "
                            f"Page {page}: "
                            f"{url}"
                        )


            assistant_message = {
                "role": "assistant",
                "content": answer,
                "source": None
            }

            st.session_state.conversations[
                selected_document
            ].append(
                assistant_message
            )


        else:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                try:

                    result = answer_question(
                        question,
                        selected_document
                    )

                    answer = result["answer"]

                    source = result.get(
                        "source"
                    )

                    st.markdown(
                        answer
                    )

                    if source:

                        with st.expander(
                            "Source Reference"
                        ):

                            st.write(
                                f"File: "
                                f"{source.get(
                                    'filename',
                                    'Unknown'
                                )}"
                            )

                            st.write(
                                f"Page: "
                                f"{source.get(
                                    'page',
                                    'Unknown'
                                )}"
                            )

                            st.write(
                                f"Section: "
                                f"{source.get(
                                    'section',
                                    'General'
                                )}"
                            )

                            st.write(
                                f"File type: "
                                f"{source.get(
                                    'file_type',
                                    'Unknown'
                                )}"
                            )

                            if source.get(
                                "sheet"
                            ):

                                st.write(
                                    f"Sheet: "
                                    f"{source['sheet']}"
                                )

                            st.write(
                                f"Chunk: "
                                f"{source.get(
                                    'chunk_id',
                                    'Unknown'
                                )}"
                            )

                            if (
                                source.get(
                                    "distance"
                                )
                                is not None
                            ):

                                st.write(
                                    f"Distance: "
                                    f"{source['distance']:.4f}"
                                )

                    elif (
                        answer
                        == NOT_AVAILABLE_MESSAGE
                    ):

                        st.info(
                            "No relevant source found."
                        )


                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "source": source
                    }

                    st.session_state.conversations[
                        selected_document
                    ].append(
                        assistant_message
                    )


                except Exception as error:

                    error_message = (
                        f"Error generating answer: "
                        f"{error}"
                    )

                    st.error(
                        error_message
                    )

                    assistant_message = {
                        "role": "assistant",
                        "content": error_message,
                        "source": None
                    }

                    st.session_state.conversations[
                        selected_document
                    ].append(
                        assistant_message
                    )