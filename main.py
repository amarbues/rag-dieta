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


def find_step() -> None:
    ingestor: DocumentIngestor = state["ingestor"]
    state["found"] = ingestor.get_new_chunks()
    st.write(f"Found {len(state['found'][0])} new chunks")


def embed_step() -> None:
    ingestor: DocumentIngestor = state["ingestor"]
    new_chunks, new_ids = state["found"]
    state["retriever"] = ingestor.embed_documents(new_chunks, new_ids)


def retrieval_step() -> None:
    retriever: VectorStoreRetriever = state["retriever"]
    state["results"] = retriever.invoke(prompt)


steps = [
    ("Ingest", ingest_step),
    ("Find", find_step),
    ("Embed", embed_step),
    ("Retrieve", retrieval_step),
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
        elapsed = time.gmtime(time.time() - st.session_state.start_time)
        st.write(f"{label}: {elapsed.tm_min}m {elapsed.tm_sec}s")
        st.divider()

    if "results" in state:
        st.write(state["results"])
