import time
from typing import Any

import streamlit as st
from langchain.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.ingest import Ingestor, Loader
from app.rag import LLM, RAG

LOADER = PyPDFLoader
LOADER_KWARGS = {"extraction_mode": "layout"}
EMBEDDING_MODEL = OllamaEmbeddings(model="qwen3-embedding:4b")
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20
LLM_MODEL = OllamaLLM(model="qwen3.5:9b")
K = 6


class Pipeline:
    """Encapsulates the RAG pipeline state and operations."""

    def __init__(
        self,
        loader: type[Loader],
        loader_kwargs: dict[str, Any],
        embedding_model: Embeddings,
        chunk_size: int,
        chunk_overlap: int,
        llm_model: LLM,
        k: int,
        prompt: str,
    ):
        # init variables
        self.loader = loader
        self.loader_kwargs = loader_kwargs
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model = llm_model
        self.k = k
        self.prompt = prompt

        # pipeline state
        self.ingestor: Ingestor | None = None
        self.discovered_chunks: tuple[list[Document], list[str]] | None = None
        self.vectorstore: Chroma | None = None
        self.rag: RAG | None = None
        self.retrieved_chunks: list[Document] | None = None
        self.formatted_prompt: str | None = None
        self.chat_response: str | Exception = ""

    def __ingest(self) -> None:
        """Load and chunk PDFs."""
        self.ingestor = Ingestor(
            loader=self.loader,
            loader_kwargs=self.loader_kwargs,
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = self.ingestor.ingest_pdf()
        st.sidebar.code(
            chunks,
            wrap_lines=True,
            height=300,
        )

    def __discover(self) -> None:
        """Discover new chunks not yet in vectorstore."""
        if self.ingestor is None:
            raise ValueError("Must run ingest() first")

        self.discovered_chunks = self.ingestor.discover_chunks()
        st.sidebar.write(f"Discovered {len(self.discovered_chunks[0])} new chunks")

    def __embed(self) -> None:
        """Embed and store new chunks."""
        if self.ingestor is None or self.discovered_chunks is None:
            raise ValueError("Must run ingest() and discover() first")

        new_chunks, new_ids = self.discovered_chunks
        self.vectorstore = self.ingestor.embed_new_chunks(new_chunks, new_ids)

    def __retrieve(self) -> None:
        """Retrieve relevant chunks for the prompt."""
        if self.vectorstore is None:
            raise ValueError("Must run embed() first")

        self.rag = RAG(
            vectorstore=self.vectorstore,
            llm=self.llm_model,
        )
        self.retrieved_chunks = self.rag.retrieve_chunks(self.prompt, self.k)

        st.sidebar.write(f"Retrieved {len(self.retrieved_chunks)} chunks")
        st.sidebar.code(
            self.retrieved_chunks,
            wrap_lines=True,
            height=300,
        )

    def __prepare(self) -> None:
        """Format prompt with context."""
        if self.rag is None:
            raise ValueError("Must run retrieve() first")

        self.formatted_prompt = self.rag.prepare_prompt(self.prompt)

        st.sidebar.text_area(
            "Prompt",
            value=self.formatted_prompt,
            height=300,
            disabled=True,
        )

    def __respond(self) -> None:
        """Generate response based on retrieved chunks."""
        if self.rag is None or self.formatted_prompt is None:
            raise ValueError("Must run retrieve() and prepare() first")

        try:
            self.chat_response = self.rag.generate_response()
        except Exception as e:
            self.chat_response = e

    def run(self) -> None:
        """Execute the full pipeline."""
        steps = [
            ("Ingest", self.__ingest),
            ("Discover", self.__discover),
            ("Embed", self.__embed),
            ("Retrieve", self.__retrieve),
            ("Prepare", self.__prepare),
            ("Respond", self.__respond),
        ]

        for label, func in steps:
            start_time = time.time()
            with st.sidebar.spinner(f" {label}..."):
                func()
            elapsed = time.gmtime(time.time() - start_time)
            st.sidebar.caption(f"{label}: {elapsed.tm_min}m {elapsed.tm_sec}s")
            st.sidebar.divider()


# frontend logic #####################################################################################

st.set_page_config(
    layout="wide",
)

st.sidebar.header("Status")
st.title("RAG Dieta")

prompt = st.text_area(
    "Fammi una domanda sulla tua dieta",
    height="stretch",
    value="Quante verdure posso mangiare?",
)

st.session_state.pipeline = Pipeline(
    loader=LOADER,
    loader_kwargs=LOADER_KWARGS,
    embedding_model=EMBEDDING_MODEL,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    llm_model=LLM_MODEL,
    k=K,
    prompt=prompt,
)

if st.button("Invia"):
    with st.spinner():
        st.session_state.pipeline.run()

st.write(st.session_state.pipeline.chat_response)
