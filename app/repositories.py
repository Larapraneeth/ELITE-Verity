from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import Document


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
