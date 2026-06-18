from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

PROMPT_TEMPLATE = """
Domanda:
{prompt}

Usa il contesto per rispondere alla domanda.
Se il contesto non contiene la risposta, rispondi solo con "Non lo so.".

Contesto:
{context}"""


class LLM(Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def invoke(self, *args: Any, **kwargs: Any) -> Any: ...


class RAG:
    def __init__(
        self,
        vectorstore_retriever: VectorStoreRetriever,
        llm: LLM,
    ) -> None:
        self.vectorstore_retriever = vectorstore_retriever
        self.llm = llm
        self.retrieved_chunks: list[Document]

    def retrieve_chunks(self, prompt: str) -> list[Document]:
        retrieved_chunks = self.vectorstore_retriever.invoke(prompt)
        self.retrieved_chunks = retrieved_chunks
        return retrieved_chunks

    def generate_response(self, prompt: str) -> str:
        context = [d.page_content for d in self.retrieved_chunks]
        prompt = PROMPT_TEMPLATE.format(prompt=prompt, context=context)
        # response = self.llm.invoke(prompt)
        response = "ciao vai \n\n" + prompt
        return response
