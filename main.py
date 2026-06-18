import time

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.ingest import Ingestor
from app.rag import RAG


class Pipeline:
    """Encapsulates the RAG pipeline state and operations."""

    def __init__(self):
        self.ingestor: Ingestor | None = None
        self.discovered_chunks: tuple[list[Document], list[str]] | None = None
        self.vectorstore_retriever = None
        self.rag: RAG | None = None
        self.retrieved_chunks: list[Document] | None = None
        self.chat_response: str | Exception = ""

    def __ingest(self) -> None:
        """Load and chunk PDFs."""
        self.ingestor = Ingestor(
            loader=PyPDFLoader,
            loader_kwargs={"extraction_mode": "layout"},
            embedding_model=OllamaEmbeddings(model="qwen3-embedding:4b"),
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

    def __retrieve(self, prompt: str) -> None:
        """Retrieve relevant chunks for the prompt."""
        if self.vectorstore_retriever is None:
            raise ValueError("Must run embed() first")

        self.rag = RAG(
            vectorstore_retriever=self.vectorstore_retriever,
            llm=OllamaLLM(model="qwen3.5:9b"),
        )
        self.retrieved_chunks = self.rag.retrieve_chunks(prompt)

        st.sidebar.write(f"Retrieved {len(self.retrieved_chunks)} chunks")
        st.sidebar.code(
            self.retrieved_chunks,
            wrap_lines=True,
            height=300,
        )

    def __respond(self, prompt: str) -> None:
        """Generate response based on retrieved chunks."""
        if self.rag is None:
            raise ValueError("Must run retrieve() first")

        try:
            self.chat_response = self.rag.generate_response(prompt)
        except Exception as e:
            self.chat_response = e

    def run(self, prompt: str) -> None:
        """Execute the full pipeline."""
        steps = [
            ("Ingest", self.__ingest),
            ("Discover", self.__discover),
            ("Embed", self.__embed),
            ("Retrieve", lambda: self.__retrieve(prompt)),
            ("Respond", lambda: self.__respond(prompt)),
        ]

        for label, func in steps:
            start_time = time.time()
            with st.sidebar.spinner(f" {label}..."):
                func()
            elapsed = time.gmtime(time.time() - start_time)
            st.sidebar.caption(f"{label}: {elapsed.tm_min}m {elapsed.tm_sec}s")
            st.sidebar.divider()


# frontend logic #####################################################################################

st.session_state.pipeline = Pipeline()

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

if st.button("Invia"):
    with st.spinner():
        st.session_state.pipeline.run(prompt)

st.write(st.session_state.pipeline.chat_response)
