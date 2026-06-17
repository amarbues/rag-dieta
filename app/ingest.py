from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Loader(Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def load(self) -> list[Document]: ...


class DocumentIngestor:
    def __init__(
        self,
        loader: type[Loader],
        loader_kwargs: dict[str, Any] | None = None,
    ):
        self.path = Path("documents")
        self.loader = loader
        self.loader_kwargs = loader_kwargs or {}

    def ingest_pdf(self):
        # load all documents
        all_docs: list[Document] = []
        for pdf_file in self.path.glob("*.pdf"):
            docs = self.loader(
                str(pdf_file),
                **self.loader_kwargs,
            ).load()
            all_docs.extend(docs)

        # split documents into chunks
        chunks = RecursiveCharacterTextSplitter().split_documents(all_docs)

        return chunks
