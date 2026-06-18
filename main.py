import time
from typing import Any

import streamlit as st
from langchain.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.ingest import Ingestor, Loader
from app.rag import LLM, RAG

LOADER = PyPDFLoader
LOADER_KWARGS = {"extraction_mode": "layout"}
EMBEDDING_MODEL = OllamaEmbeddings(model="qwen3-embedding:4b")
LLM_MODEL = OllamaLLM(model="qwen3.5:9b")


class Pipeline:
    """Encapsulates the RAG pipeline state and operations."""

    def __init__(
        self,
        loader: type[Loader],
        loader_kwargs: dict[str, Any],
        embedding_model: Embeddings,
        llm_model: LLM,
        prompt: str,
    ):
        # init variables
        self.loader = loader
        self.loader_kwargs = loader_kwargs
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.prompt = prompt

        # pipeline state
        self.ingestor: Ingestor | None = None
        self.discovered_chunks: tuple[list[Document], list[str]] | None = None
        self.vectorstore_retriever = None
        self.rag: RAG | None = None
        self.retrieved_chunks: list[Document] | None = None
        self.chat_response: str | Exception = ""

    def __ingest(self) -> None:
        """Load and chunk PDFs."""
        self.ingestor = Ingestor(
            loader=self.loader,
            loader_kwargs=self.loader_kwargs,
            embedding_model=self.embedding_model,
        )
        self.ingestor.ingest_pdf()

    def __discover(self) -> None:
        """Discover new chunks not yet in vectorstore."""
        if self.ingestor is None:
            raise ValueError("Must run ingest() first")

        self.discovered_chunks = self.ingestor.get_new_chunks()
        st.sidebar.write(f"Discovered {len(self.discovered_chunks[0])} new chunks")

    def __embed(self) -> None:
        """Embed and store new chunks."""
        if self.ingestor is None or self.discovered_chunks is None:
            raise ValueError("Must run ingest() and discover() first")

        new_chunks, new_ids = self.discovered_chunks
        self.vectorstore_retriever = self.ingestor.embed_documents(new_chunks, new_ids)

    def __retrieve(self) -> None:
        """Retrieve relevant chunks for the prompt."""
        if self.vectorstore_retriever is None:
            raise ValueError("Must run embed() first")

        self.rag = RAG(
            vectorstore_retriever=self.vectorstore_retriever,
            llm=self.llm_model,
        )
        self.retrieved_chunks = self.rag.retrieve_chunks(self.prompt)

        st.sidebar.write(f"Retrieved {len(self.retrieved_chunks)} chunks")
        st.sidebar.code(
            self.retrieved_chunks,
            wrap_lines=True,
            height=300,
        )

    def __respond(self) -> None:
        """Generate response based on retrieved chunks."""
        if self.rag is None:
            raise ValueError("Must run retrieve() first")

        try:
            self.chat_response = self.rag.generate_response(self.prompt)
        except Exception as e:
            self.chat_response = e

    def run(self) -> None:
        """Execute the full pipeline."""
        steps = [
            ("Ingest", self.__ingest),
            ("Discover", self.__discover),
            ("Embed", self.__embed),
            ("Retrieve", self.__retrieve),
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
    llm_model=LLM_MODEL,
    prompt=prompt,
)

if st.button("Invia"):
    with st.spinner():
        st.session_state.pipeline.run()

st.write(st.session_state.pipeline.chat_response)
