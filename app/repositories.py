from sqlalchemy import select
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
