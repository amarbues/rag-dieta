import datetime
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
        self.raw_prompt = prompt

        # pipeline state
        self.rewritten_prompt: str | None = None
        self.retrieved_chunks: list[Document] | None = None
        self.formatted_prompt: str | None = None

    def rewrite_prompt(self) -> str:
        weekdays = [
            "lunedi",
            "martedi",
            "mercoledi",
            "giovedi",
            "venerdi",
            "sabato",
            "domenica",
        ]
        today = datetime.date.today()
        weekday = weekdays[today.weekday()]
        self.rewritten_prompt = self.raw_prompt.lower().replace("oggi", weekday)
        return self.rewritten_prompt

    def retrieve_chunks(self) -> list[Document]:
        if not self.rewritten_prompt:
            raise ValueError("You must call rewrite_prompt() first")

        docs_and_scores = self.vectorstore.similarity_search_with_score(
            self.rewritten_prompt,
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
        if not self.retrieved_chunks:
            raise ValueError("You must call retrieve_chunks() first")

        # 1. Extract the parent content instead of the small chunk
        # 2. Use a set to deduplicate in case multiple chunks point to the same page
        unique_parents: set[str] = set()
        for doc in self.retrieved_chunks:
            unique_parents.add(doc.metadata["parent_content"])

        # 3. Join the unique parent pages
        context_string = "\n\n--- NUOVA PAGINA ---\n\n".join(unique_parents)

        self.formatted_prompt = PROMPT_TEMPLATE.format(
            prompt=self.rewritten_prompt,
            context=context_string,
        )
        return self.formatted_prompt

    def generate_response(self) -> str:
        if not self.formatted_prompt:
            raise ValueError("You must call prepare_prompt() first")

        response = self.llm.invoke(self.formatted_prompt)
        return response
