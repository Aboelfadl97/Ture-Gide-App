"""
07_prompting.py

Generate answers using Groq and retrieved
context from the Digital Notes PDF.
"""

import os
from importlib import import_module
from typing import List, Tuple

from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


retrieval_module = import_module(
    "06_retrieve_context"
)

retrieve_context = (
    retrieval_module.retrieve_context
)


PROMPT_TEMPLATE = """
You are True Guide, a Data Analytics study assistant.

Answer the user's question using ONLY the provided context
from the Digital Notes PDF.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the context, say:
   "I could not find this information in the Digital Notes."
4. Explain the answer clearly and simply.
5. Keep the answer focused and organized.
6. Do not invent page numbers or sources.

Retrieved context:
{context}

Question:
{question}

Answer:
"""


def get_page_number(
    document: Document
):
    """
    Return a human-readable PDF page number.
    """

    page_number = document.metadata.get(
        "page_number"
    )

    if page_number is not None:
        return page_number

    original_page = document.metadata.get(
        "page"
    )

    if original_page is not None:
        return int(original_page) + 1

    return "Unknown"


def build_context(
    documents: List[Document]
) -> str:
    """
    Convert retrieved documents into
    a formatted context string.
    """

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):
        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page_number = get_page_number(
            document
        )

        context_part = (
            f"[Context {index}]\n"
            f"Source: {source}\n"
            f"Page: {page_number}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

        context_parts.append(
            context_part
        )

    return "\n\n".join(
        context_parts
    )


def get_sources(
    documents: List[Document]
) -> List[str]:
    """
    Return unique source and page references.
    """

    sources = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page_number = get_page_number(
            document
        )

        source_reference = (
            f"{source}, page {page_number}"
        )

        if source_reference not in sources:
            sources.append(
                source_reference
            )

    return sources


def generate_answer(
    question: str,
    k: int = 4
) -> Tuple[str, List[str]]:
    """
    Retrieve context and generate an answer
    using Groq.
    """

    if not question or not question.strip():
        raise ValueError(
            "The question cannot be empty."
        )

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY was not found. "
            "Check your .env file."
        )

    documents = retrieve_context(
        query=question.strip(),
        k=k
    )

    if not documents:
        return (
            "I could not find this information "
            "in the Digital Notes.",
            []
        )

    context = build_context(
        documents
    )

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question.strip()
    )

    client = Groq(
        api_key=GROQ_API_KEY
    )

    response = (
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )
    )

    answer = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    sources = get_sources(
        documents
    )

    return answer, sources


def main():
    """
    Test the complete RAG pipeline.
    """

    try:
        question = input(
            "Ask True Guide: "
        ).strip()

        answer, sources = generate_answer(
            question=question,
            k=4
        )

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        if sources:
            for source in sources:
                print(f"- {source}")
        else:
            print("No sources were retrieved.")

    except Exception as error:
        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)
        print(type(error).__name__)
        print(error)


if __name__ == "__main__":
    main()