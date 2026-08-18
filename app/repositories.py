from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import Conversation, Document, Message


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, filename: str, file_type: str, status: str = "pending") -> Document:
        document = Document(
            filename=filename,
            file_type=file_type,
            status=status,
        )
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document

    def list(self) -> list[Document]:
        statement = select(Document).order_by(Document.created_at.desc(), Document.id.desc())
        return list(self.session.scalars(statement))

    def get(self, document_id: int) -> Document | None:
        return self.session.get(Document, document_id)

    def get_by_filename(self, filename: str) -> Document | None:
        statement = select(Document).where(Document.filename == filename)
        return self.session.scalars(statement).first()

    def get_or_create_by_filename(self, filename: str) -> Document:
        """Return the Document row for ``filename``, creating it if missing.

        Uses the unique ``Document.filename`` constraint, so repeated calls
        (or concurrent ingestion) can never create duplicate rows.
        """
        document = self.get_by_filename(filename)
        if document is not None:
            return document

        file_type = Path(filename).suffix.lstrip(".").lower() or "unknown"
        document = Document(
            filename=filename,
            file_type=file_type,
            status="ready",
        )
        self.session.add(document)
        try:
            self.session.commit()
        except SQLAlchemyError:
            self.session.rollback()
            document = self.get_by_filename(filename)
            if document is None:
                raise
        self.session.refresh(document)
        return document

    def update_status(self, document_id: int, status: str) -> Document | None:
        document = self.get(document_id)
        if document is None:
            return None

        document.status = status
        self.session.commit()
        self.session.refresh(document)
        return document

    def list_pending(self) -> list[Document]:
        statement = select(Document).where(Document.status == "pending")
        return list(self.session.scalars(statement))

    def claim_pending(self, document_id: int) -> Document | None:
        statement = (
            update(Document)
            .where(Document.id == document_id, Document.status == "pending")
            .values(status="processing")
        )
        result = self.session.execute(statement)
        self.session.commit()
        if result.rowcount != 1:
            return None
        return self.get(document_id)

    def recover_stale_processing(self, stale_seconds: int) -> int:
        """Atomically reset 'processing' documents that have been stuck for
        longer than ``stale_seconds`` back to 'pending' so the worker can pick
        them up again. Uses ``updated_at``, which is refreshed on every status
        change including the atomic ``claim_pending`` transition.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
        statement = (
            update(Document)
            .where(
                Document.status == "processing",
                Document.updated_at < cutoff,
            )
            .values(status="pending")
        )
        result = self.session.execute(statement)
        self.session.commit()
        return result.rowcount


class ConversationRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_or_create(
        self,
        document_id: int,
        conversation_id: int | None,
    ) -> Conversation:
        """Reuse an existing conversation for the document, or create a new one.

        A supplied ``conversation_id`` is reused only when it belongs to the
        requested document; an invalid or mismatched id falls back to a fresh
        conversation for the requested document.
        """
        if conversation_id is not None:
            conversation = self.session.get(Conversation, conversation_id)
            if (
                conversation is not None
                and conversation.document_id == document_id
            ):
                return conversation

        conversation = Conversation(document_id=document_id)
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)
        return message
