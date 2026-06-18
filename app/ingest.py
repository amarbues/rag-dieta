from pathlib import Path
from typing import Any, Protocol

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
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

        self.chunks.extend(chunks)

    def embed_documents(self) -> VectorStoreRetriever:
        embeddings = OllamaEmbeddings(
            model="qwen3-embedding:4b",
            dimensions=4096,
        )

        vector_store = InMemoryVectorStore.from_documents(
            documents=self.chunks,
            embedding=embeddings,
        )

        return vector_store.as_retriever()


if __name__ == "__main__":
    ingestor = DocumentIngestor(
        loader=PyPDFLoader,
        loader_kwargs={"extraction_mode": "layout"},
    )
    ingestor.ingest_pdf()
    retriever = ingestor.embed_documents()
    response = retriever.invoke("Quante verdure posso mangiare?")
    print(response)
