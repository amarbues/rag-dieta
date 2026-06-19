import hashlib
import re
from pathlib import Path
from typing import Any, Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Loader(Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def load(self) -> list[Document]: ...


class Ingestor:
    def __init__(
        self,
        loader: type[Loader],
        loader_kwargs: dict[str, Any],
        vectorstore: Chroma,
        chunk_size: int,
        chunk_overlap: int,
    ):
        # init globals
        self.path = Path("documents")

        # init variables
        self.loader = loader
        self.loader_kwargs = loader_kwargs
        self.vectorstore = vectorstore
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # pipeline state
        self.ingested_chunks: list[Document]
        self.discovered_chunks: list[Document]
        self.discovered_ids: list[str]

    def ingest_pdf(self) -> list[Document]:
        ingested_chunks: list[Document] = []

        # process all pages
        for pdf_file in self.path.glob("*.pdf"):
            raw_pages = self.loader(
                str(pdf_file),
                **self.loader_kwargs,
            ).load()

            for rp in raw_pages:
                # Try to identify the identifier on the page
                day_match = re.search(
                    r"(Lunedi|Martedi|Mercoledi|Giovedi|Venerdi|Sabato|Domenica)",
                    rp.page_content,
                )
                section_match = re.search(
                    r"(SOSTITUTI ALIMENTI|LINEE GUIDA|FAQ|TABELLA CONVERSIONE)",
                    rp.page_content,
                )
                page_header = ""
                if day_match:
                    page_header = f"Giorno: {day_match.group(1)}"
                elif section_match:
                    page_header = f"Sezione: {section_match.group(1)}"
                else:
                    page_header = f"Pagina {rp.metadata['page_label']}"

                # split each page into chunks
                child_chunks = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ).split_documents([rp])

                for i, chunk in enumerate(child_chunks):
                    # Prepend the header so the embedding model understands the temporal context
                    chunk.page_content = f"[{page_header}] {chunk.page_content}"

                    # Pack all tracking parameters and the entire parent page into metadata
                    chunk.metadata["header"] = page_header
                    chunk.metadata["chunk_index"] = i
                    chunk.metadata["parent_content"] = rp.page_content

                    ingested_chunks.append(chunk)

        self.ingested_chunks = ingested_chunks
        return ingested_chunks

    def discover_chunks(self) -> int:
        # get vectorstore and existing row ids
        existing = self.vectorstore.get()
        existing_ids: set[str] = set(existing["ids"]) if existing["ids"] else set()

        # check if there are new chunks to be added to the vectorstore
        discovered_chunks: list[Document] = []
        discovered_ids: list[str] = []
        for doc in self.ingested_chunks:
            cid = self.__generate_chunk_id(doc)
            if cid not in existing_ids and cid not in discovered_ids:
                discovered_chunks.append(doc)
                discovered_ids.append(cid)

        self.discovered_chunks = discovered_chunks
        self.discovered_ids = discovered_ids
        return len(discovered_chunks)

    def embed_new_chunks(self) -> None:
        if self.discovered_chunks:
            self.vectorstore.add_documents(
                self.discovered_chunks,
                ids=self.discovered_ids,
            )

    def __generate_chunk_id(self, doc: Document) -> str:
        """Generate a unique id for each chunk based on its content"""
        source = doc.metadata.get("source", "unknown")
        header = doc.metadata.get("header", "unknown")
        idx = doc.metadata.get("chunk_index", 0)

        # Create a unique string combining geography and content
        unique_str = f"{source}_{header}_c{idx}_{doc.page_content}"
        return hashlib.md5(unique_str.encode()).hexdigest()
