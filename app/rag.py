from typing import Any, Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document

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
        vectorstore: Chroma,
        k: int,
        llm: LLM,
        prompt: str,
    ) -> None:
        # init variables
        self.vectorstore = vectorstore
        self.k = k
        self.llm = llm
        self.prompt = prompt

        # pipeline state
        self.retrieved_chunks: list[Document]
        self.formatted_prompt: str

    def retrieve_chunks(self) -> list[Document]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            self.prompt,
            k=self.k,
        )

        self.retrieved_chunks = []
        seen_content: set[str] = set()
        for doc, score in docs_and_scores:
            # Inject the score into the metadata
            doc.metadata["similarity_score"] = score

            # dedup page_content, so we don't blow up context
            if doc.page_content not in seen_content:
                self.retrieved_chunks.append(doc)
            seen_content.add(doc.page_content)

        return self.retrieved_chunks

    def prepare_prompt(self) -> str:
        # 1. Extract the parent content instead of the small chunk
        # 2. Use a set to deduplicate in case multiple chunks point to the same page
        unique_parents: set[str] = set()
        for doc in self.retrieved_chunks:
            unique_parents.add(doc.metadata["parent_content"])

        # 3. Join the unique parent pages
        context_string = "\n\n--- NUOVA PAGINA ---\n\n".join(unique_parents)

        self.formatted_prompt = PROMPT_TEMPLATE.format(
            prompt=self.prompt, context=context_string
        )
        return self.formatted_prompt

    def generate_response(self) -> str:
        response = self.llm.invoke(self.formatted_prompt)
        return response
