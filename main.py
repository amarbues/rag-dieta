import streamlit as st


st.title("RAG Dieta")

prompt = st.text_area(
    "Fammi una domanda sulla tua dieta",
    height="stretch",
    value="Cosa posso mangiare oggi a pranzo?",
)

if st.button("Invia"):
    st.write(f'Hai chiesto: "{prompt.strip()}"')
