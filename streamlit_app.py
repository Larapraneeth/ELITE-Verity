import os

import httpx
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


API_BASE_URL = os.environ.get(
    "RAGDOC_API_URL",
    "http://localhost:8000"
)


def _api_get(path):

    with httpx.Client(timeout=30.0) as client:

        response = client.get(
            f"{API_BASE_URL}{path}"
        )

        response.raise_for_status()

        return response.json()


def _api_post(path, payload):

    with httpx.Client(timeout=120.0) as client:

        response = client.post(
            f"{API_BASE_URL}{path}",
            json=payload
        )

        response.raise_for_status()

        return response.json()


def fetch_documents():

    return _api_get("/api/documents")


def find_document_id(filename):

    for document in fetch_documents():

        if document.get("filename") == filename:

            return document.get("id")

    return None


def fetch_conversations(document_id):

    return _api_get(
        f"/api/documents/{document_id}/conversations"
    )


def fetch_messages(document_id, conversation_id):

    return _api_get(
        f"/api/documents/{document_id}/conversations/"
        f"{conversation_id}/messages"
    )


def _to_display_messages(api_messages):

    display = []

    for message in api_messages:

        display.append(
            {
                "role": message.get("role"),
                "content": message.get("content"),
                "source": message.get("source"),
                "created_at": message.get("created_at")
            }
        )

    return display


def _sorted_conversations(conversations):

    return sorted(
        conversations,
        key=lambda conv: (
            conv.get("last_message_created_at") or "",
            conv.get("id") or 0
        ),
        reverse=True
    )


def _format_timestamp(iso_value):

    if not iso_value:

        return ""

    try:

        from datetime import datetime

        parsed = datetime.fromisoformat(
            iso_value
        )

        return parsed.strftime("%Y-%m-%d %H:%M")

    except Exception:

        return iso_value


def _format_display_datetime(iso_value):

    if not iso_value:

        return None

    try:

        from datetime import datetime
        from zoneinfo import ZoneInfo

        parsed = datetime.fromisoformat(
            str(iso_value).replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:

            parsed = parsed.replace(
                tzinfo=ZoneInfo("UTC")
            )

        local_time = parsed.astimezone(
            ZoneInfo("Asia/Kolkata")
        )

        month = local_time.strftime("%b")

        day = local_time.day

        year = local_time.year

        hour12 = local_time.hour % 12

        if hour12 == 0:

            hour12 = 12

        ampm = "AM" if local_time.hour < 12 else "PM"

        minute = f"{local_time.minute:02d}"

        return (
            f"{month} {day}, {year} \u2022 "
            f"{hour12:02d}:{minute} {ampm}"
        )

    except Exception:

        return None


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

if "active_conversation_ids" not in st.session_state:
    st.session_state.active_conversation_ids = {}

if "loaded_conversation_ids" not in st.session_state:
    st.session_state.loaded_conversation_ids = {}

if "available_conversations" not in st.session_state:
    st.session_state.available_conversations = {}

if "history_initialized_docs" not in st.session_state:
    st.session_state.history_initialized_docs = set()

if "document_ids" not in st.session_state:
    st.session_state.document_ids = {}


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


# --- Persisted history via FastAPI (never touches SQLite directly) ---

def _get_document_id(filename):

    if filename in st.session_state.document_ids:

        return st.session_state.document_ids[filename]

    document_id = find_document_id(filename)

    if document_id is not None:

        st.session_state.document_ids[filename] = document_id

    return document_id


def _auto_load_history(doc_name, doc_id):

    conversations = fetch_conversations(doc_id)

    ordered = _sorted_conversations(conversations)

    st.session_state.available_conversations[doc_name] = ordered

    st.session_state.loaded_conversation_ids[doc_name] = []

    if not ordered:

        st.session_state.active_conversation_ids[doc_name] = None

        st.session_state.conversations[doc_name] = []

        return

    latest = ordered[0]

    messages = fetch_messages(
        doc_id,
        latest["id"]
    )

    st.session_state.conversations[doc_name] = (
        _to_display_messages(messages)
    )

    st.session_state.active_conversation_ids[
        doc_name
    ] = latest["id"]

    st.session_state.loaded_conversation_ids[
        doc_name
    ] = [latest["id"]]


def _load_older_history(doc_name, doc_id):

    loaded_ids = st.session_state.loaded_conversation_ids.get(
        doc_name,
        []
    )

    loaded_set = set(loaded_ids)

    older = None

    for conv in st.session_state.available_conversations.get(
        doc_name,
        []
    ):

        if conv["id"] not in loaded_set:

            older = conv
            break

    if older is None:

        return False

    older_messages = _to_display_messages(
        fetch_messages(
            doc_id,
            older["id"]
        )
    )

    current = st.session_state.conversations.get(
        doc_name,
        []
    )

    st.session_state.conversations[doc_name] = (
        older_messages + current
    )

    st.session_state.loaded_conversation_ids[
        doc_name
    ] = loaded_ids + [older["id"]]

    return True


def _on_conversation_selected():

    doc_name = st.session_state.selected_document

    selector_key = st.session_state.get(
        "_active_selector_key",
        "conversation_selector"
    )

    selected = st.session_state.get(selector_key)

    is_empty = (
        selected is None
        or selected == "New conversation"
        or (
            isinstance(selected, str)
            and not selected.strip()
        )
    )

    if is_empty:

        st.session_state.active_conversation_ids[
            doc_name
        ] = None

        st.session_state.conversations[doc_name] = []

        st.session_state.loaded_conversation_ids[
            doc_name
        ] = []

        return

    doc_id = st.session_state.document_ids.get(doc_name)

    if doc_id is None:

        return

    conv_id = None

    for conversation in st.session_state.available_conversations.get(
        doc_name,
        []
    ):

        label = f"Conversation {conversation.get('id')}"

        timestamp = _format_timestamp(
            conversation.get("last_message_created_at")
        )

        if timestamp:

            label += f"  ({timestamp})"

        if label == selected:

            conv_id = conversation.get("id")

            break

    if conv_id is None:

        st.session_state.active_conversation_ids[
            doc_name
        ] = None

        st.session_state.conversations[doc_name] = []

        st.session_state.loaded_conversation_ids[
            doc_name
        ] = []

        return

    try:

        messages = fetch_messages(doc_id, conv_id)

    except Exception:

        st.warning(
            "This conversation is no longer available. "
            "Starting a new conversation."
        )

        st.session_state.active_conversation_ids[
            doc_name
        ] = None

        st.session_state.conversations[doc_name] = []

        st.session_state.loaded_conversation_ids[
            doc_name
        ] = []

        return

    st.session_state.conversations[doc_name] = (
        _to_display_messages(messages)
    )

    st.session_state.active_conversation_ids[
        doc_name
    ] = conv_id

    st.session_state.loaded_conversation_ids[
        doc_name
    ] = [conv_id]


document_id = _get_document_id(selected_document)


if document_id is not None:

    if (
        selected_document
        not in st.session_state.history_initialized_docs
    ):

        try:

            _auto_load_history(
                selected_document,
                document_id
            )

        except Exception:

            pass

        st.session_state.history_initialized_docs.add(
            selected_document
        )


available_conversations = (
    st.session_state.available_conversations.get(
        selected_document,
        []
    )
)


st.sidebar.subheader(
    "Conversation History"
)


if st.sidebar.button(
    "New conversation",
    use_container_width=True
):

    st.session_state.active_conversation_ids[
        selected_document
    ] = None

    st.session_state.conversations[
        selected_document
    ] = []

    st.session_state.loaded_conversation_ids[
        selected_document
    ] = []


if available_conversations:

    conversation_options = ["New conversation"]

    for conv in available_conversations:

        label = f"Conversation {conv.get('id')}"

        timestamp = _format_timestamp(
            conv.get("last_message_created_at")
        )

        if timestamp:

            label += f"  ({timestamp})"

        conversation_options.append(label)

    active_id = st.session_state.active_conversation_ids.get(
        selected_document
    )

    label_to_id = {
        "New conversation": None
    }

    for conv, label in zip(
        available_conversations,
        conversation_options[1:]
    ):

        label_to_id[label] = conv.get("id")

    current_index = 0

    if active_id is not None:

        for idx, label in enumerate(
            conversation_options
        ):

            if label_to_id.get(label) == active_id:

                current_index = idx
                break

    selector_key = (
        f"conversation_selector_{selected_document}"
    )

    st.session_state._active_selector_key = (
        selector_key
    )

    st.sidebar.selectbox(
        "Previous conversations",
        conversation_options,
        index=current_index,
        key=selector_key,
        on_change=_on_conversation_selected
    )


st.sidebar.caption(
    "History is loaded from the ELITE Verity API."
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


loaded_count = len(
    st.session_state.loaded_conversation_ids.get(
        selected_document,
        []
    )
)

available_count = len(
    st.session_state.available_conversations.get(
        selected_document,
        []
    )
)

if (
    document_id is not None
    and available_count > loaded_count
):

    if st.button(
        "Load older history"
    ):

        try:

            _load_older_history(
                selected_document,
                document_id
            )

        except Exception as error:

            st.error(
                f"Could not load older history: {error}"
            )

elif available_count > 0:

    st.caption(
        "Viewing the full history for this document."
    )


messages = st.session_state.conversations[
    selected_document
]


for message in messages:

    with st.chat_message(
        message["role"]
    ):

        created_at = _format_display_datetime(
            message.get("created_at")
        )

        if created_at:

            st.caption(
                created_at
            )

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

                    active_conversation_id = (
                        st.session_state.active_conversation_ids.get(
                            selected_document
                        )
                    )

                    payload = {
                        "question": question,
                        "document": selected_document,
                        "conversation_id": active_conversation_id
                    }

                    result = _api_post(
                        "/api/chat",
                        payload
                    )

                    answer = result["answer"]

                    source = result.get(
                        "source"
                    )

                    returned_id = result.get(
                        "conversation_id"
                    )

                    if returned_id is not None:

                        st.session_state.active_conversation_ids[
                            selected_document
                        ] = returned_id

                    elif active_conversation_id is not None:

                        st.session_state.active_conversation_ids[
                            selected_document
                        ] = active_conversation_id

                    elif document_id is not None:

                        try:

                            fresh = fetch_conversations(
                                document_id
                            )

                            fresh_order = _sorted_conversations(
                                fresh
                            )

                            st.session_state.available_conversations[
                                selected_document
                            ] = fresh_order

                            if fresh_order:

                                st.session_state.active_conversation_ids[
                                    selected_document
                                ] = fresh_order[0]["id"]

                        except Exception:

                            pass

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