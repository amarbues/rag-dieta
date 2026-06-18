import time
from typing import Any, Literal

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.ingest import Ingestor
from app.rag import RAG

StateObject = Literal[
    "ingestor",
    "discovered_chunks",
    "vectorstore_retriever",
    "rag",
    "retrieved_chunks",
    "chat_response",
]

# steps logic
state: dict[StateObject, Any] = {}


def ingest_step() -> None:
    state["ingestor"] = Ingestor(
        loader=PyPDFLoader,
        loader_kwargs={"extraction_mode": "layout"},
        embedding_model=OllamaEmbeddings(model="qwen3-embedding:4b"),
    )
    state["ingestor"].ingest_pdf()


def discovery_step() -> None:
    ingestor: Ingestor = state["ingestor"]
    discovered_chunks = ingestor.get_new_chunks()
    state["discovered_chunks"] = discovered_chunks
    st.sidebar.write(f"Discovered {len(discovered_chunks[0])} new chunks")


def embed_step() -> None:
    ingestor: Ingestor = state["ingestor"]
    new_chunks, new_ids = state["discovered_chunks"]
    state["vectorstore_retriever"] = ingestor.embed_documents(new_chunks, new_ids)


def retrieve_step() -> None:
    rag = RAG(
        vectorstore_retriever=state["vectorstore_retriever"],
        llm=OllamaLLM(model="qwen3.5:9b"),
    )
    state["rag"] = rag

    retrieved_chunks = rag.retrieve_chunks(prompt)
    state["retrieved_chunks"] = retrieved_chunks

    st.sidebar.write(f"Retrieved {len(retrieved_chunks)} chunks")
    st.sidebar.code(
        state["retrieved_chunks"],
        wrap_lines=True,
        height=300,
    )


def response_step() -> None:
    rag = state["rag"]

    try:
        response = rag.generate_response(prompt)
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
