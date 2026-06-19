import time
from typing import Any

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.ingest import Ingestor, Loader
from app.rag import LLM, RAG

LOADER = PyPDFLoader
LOADER_KWARGS = {"extraction_mode": "layout"}
VECTORSTORE = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OllamaEmbeddings(model="qwen3-embedding:4b"),
)
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
        vectorstore: Chroma,
        chunk_size: int,
        chunk_overlap: int,
        llm_model: LLM,
        k: int,
        prompt: str,
    ):
        # init classes
        self.ingestor = Ingestor(
            loader,
            loader_kwargs,
            vectorstore,
            chunk_size,
            chunk_overlap,
        )
        self.rag = RAG(
            vectorstore,
            k,
            llm_model,
            prompt,
        )

        # pipeline variables
        self.chat_response: str | Exception = ""

    def __ingest(self) -> None:
        """Load and chunk PDFs."""
        chunks = self.ingestor.ingest_pdf()
        st.sidebar.code(
            chunks,
            wrap_lines=True,
            height=300,
        )

    def __discover(self) -> None:
        """Discover new chunks not yet in vectorstore."""
        n_discovered_chunks = self.ingestor.discover_chunks()
        st.sidebar.write(f"Discovered {n_discovered_chunks} new chunks")

    def __embed(self) -> None:
        """Embed and store new chunks."""
        self.ingestor.embed_new_chunks()

    def __rewrite(self) -> None:
        """Rewrites prompt to be usable with context."""
        rewritten_prompt = self.rag.rewrite_prompt()
        st.sidebar.text_area(
            "Rewritten Prompt",
            value=rewritten_prompt,
            height="stretch",
            disabled=True,
        )

    def __retrieve(self) -> None:
        """Retrieve relevant chunks for the prompt."""
        retrieved_chunks = self.rag.retrieve_chunks()
        st.sidebar.write(f"Retrieved {len(retrieved_chunks)} chunks")
        st.sidebar.code(
            retrieved_chunks,
            wrap_lines=True,
            height=300,
        )

    def __prepare(self) -> None:
        """Format prompt with context."""
        formatted_prompt = self.rag.prepare_prompt()
        st.sidebar.text_area(
            "Formatted Prompt",
            value=formatted_prompt,
            height="stretch",
            disabled=True,
        )

    def __respond(self) -> None:
        """Generate response based on retrieved chunks."""
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
            ("Rewrite", self.__rewrite),
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
    LOADER,
    LOADER_KWARGS,
    VECTORSTORE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LLM_MODEL,
    K,
    prompt,
)

if st.button("Invia"):
    with st.spinner():
        st.session_state.pipeline.run()

st.write(st.session_state.pipeline.chat_response)
