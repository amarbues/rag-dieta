import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

from app.ingest import DocumentIngestor

st.title("RAG Dieta")

prompt = st.text_area(
    "Fammi una domanda sulla tua dieta",
    height="stretch",
    value="Cosa posso mangiare oggi a pranzo?",
)

if st.button("Invia"):
    with st.spinner("Running..."):
        chunks = DocumentIngestor(
            PyPDFLoader,
            loader_kwargs={"extraction_mode": "layout"},
        ).ingest_pdf()
    st.write(chunks)
