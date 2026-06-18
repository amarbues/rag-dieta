import hashlib
from pathlib import Path
from typing import Any, Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Loader(Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def load(self) -> list[Document]: ...


class DocumentIngestor:
    def __init__(
        self,
        loader: type[Loader],
        loader_kwargs: dict[str, Any] = {},
    ):
        self.path = Path("documents")
        self.loader = loader
        self.loader_kwargs = loader_kwargs
        self.chunks: list[Document] = []
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=OllamaEmbeddings(
                model="qwen3-embedding:4b",
                dimensions=4096,
            ),
        )

    def ingest_pdf(self) -> None:
        # load all documents
        all_docs: list[Document] = []
        for pdf_file in self.path.glob("*.pdf"):
            docs = self.loader(
                str(pdf_file),
                **self.loader_kwargs,
            ).load()
            all_docs.extend(docs)

        # split documents into chunks
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=4096,
            chunk_overlap=0,
        ).split_documents(all_docs)

        # return chunks
        self.chunks.extend(chunks)

    def get_new_chunks(self) -> tuple[list[Document], list[str]]:
        # get vectorstore and existing row ids
        existing = self.vectorstore.get()
        existing_ids: set[str] = set(existing["ids"]) if existing["ids"] else set()

        # check if there are new chunks to be added to the vectorstore
        new_chunks: list[Document] = []
        new_ids: list[str] = []
        for doc in self.chunks:
            cid = _generate_chunk_id(doc)
            if cid not in existing_ids:
                new_chunks.append(doc)
                new_ids.append(cid)

        # return chunks and ids
        return new_chunks, new_ids

    def embed_documents(
        self, new_chunks: list[Document], new_ids: list[str]
    ) -> VectorStoreRetriever:
        if new_chunks:
            self.vectorstore.add_documents(
                new_chunks,
                ids=new_ids,
            )
        return self.vectorstore.as_retriever()


def _generate_chunk_id(doc: Document):
    """Generate a unique id for each chunk based on its content"""
    return hashlib.md5(doc.page_content.encode()).hexdigest()
