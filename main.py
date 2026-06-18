import time
from typing import Any, Literal

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.ingest import DocumentIngestor

StateObject = Literal[
    "document_ingestor",
    "discovered_chunks",
    "vectorstore_retriever",
    "retrieved_chunks",
    "chat_response",
]

# steps logic
state: dict[StateObject, Any] = {}


def ingest_step() -> None:
    state["document_ingestor"] = DocumentIngestor(
        loader=PyPDFLoader,
        loader_kwargs={"extraction_mode": "layout"},
        embedding_model=OllamaEmbeddings(model="qwen3-embedding:4b"),
    )
    state["document_ingestor"].ingest_pdf()


def discovery_step() -> None:
    ingestor: DocumentIngestor = state["document_ingestor"]
    discovered_chunks = ingestor.get_new_chunks()
    state["discovered_chunks"] = discovered_chunks
    st.sidebar.write(f"Discovered {len(discovered_chunks[0])} new chunks")


def embed_step() -> None:
    ingestor: DocumentIngestor = state["document_ingestor"]
    new_chunks, new_ids = state["discovered_chunks"]
    state["vectorstore_retriever"] = ingestor.embed_documents(new_chunks, new_ids)


def retrieve_step() -> None:
    retriever: VectorStoreRetriever = state["vectorstore_retriever"]
    retrieved = retriever.invoke(prompt)
    st.sidebar.write(f"Retrieved {len(retrieved)} chunks")
    state["retrieved_chunks"] = retrieved


def response_step() -> None:
    retrieved: list[Document] = state["retrieved_chunks"]
    context = [d.page_content for d in retrieved]

    model = OllamaLLM(model="qwen3.5:9b")

    template = f"""
Domanda:
{prompt}

Usa il contesto per rispondere alla domanda.
Se il contesto non contiene la risposta, rispondi solo con "Non lo so.".

Contesto:
{context}"""

    st.sidebar.code(
        retrieved,
        wrap_lines=True,
        height=300,
    )

    try:
        response = model.invoke(template)
    except Exception as e:
        response = e

    state["chat_response"] = response


steps = [
    ("Ingest", ingest_step),
    ("Discover", discovery_step),
    ("Embed", embed_step),
    ("Retrieve", retrieve_step),
    ("Respond", response_step),
]

# frontend logic
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
        for label, func in steps:
            st.session_state.start_time = time.time()
            with st.sidebar.spinner(f" {label}..."):
                func()
            elapsed = time.gmtime(time.time() - st.session_state.start_time)
            st.sidebar.caption(f"{label}: {elapsed.tm_min}m {elapsed.tm_sec}s")
            st.sidebar.divider()

if "chat_response" in state:
    st.write(state["chat_response"])
