import time
from typing import Any

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import VectorStoreRetriever

from app.ingest import DocumentIngestor

# steps logic
state: dict[str, Any] = {}


def ingest_step() -> None:
    state["ingestor"] = DocumentIngestor(
        PyPDFLoader,
        loader_kwargs={"extraction_mode": "layout"},
    )
    state["ingestor"].ingest_pdf()


def embed_step() -> None:
    ingestor: DocumentIngestor = state["ingestor"]
    state["retriever"] = ingestor.embed_documents()


def retrieval_step() -> None:
    retriever: VectorStoreRetriever = state["retriever"]
    state["results"] = retriever.invoke(prompt)


steps = [
    ("Ingest", ingest_step),
    ("Embed", embed_step),
    ("Retrieval", retrieval_step),
]

# frontend logic
st.title("RAG Dieta")

prompt = st.text_area(
    "Fammi una domanda sulla tua dieta",
    height="stretch",
    value="Cosa posso mangiare oggi a pranzo?",
)

if st.button("Invia"):
    for label, func in steps:
        st.session_state.start_time = time.time()
        with st.spinner(f" {label}..."):
            func()
        elapsed = time.time() - st.session_state.start_time
        st.write(f"{label}: {elapsed:.2f} s")

    if "results" in state:
        st.write(state["results"])
