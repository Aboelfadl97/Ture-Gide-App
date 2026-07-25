"""
06_retrieve_context.py

Retrieve relevant context from ChromaDB.
If the database does not exist, create it automatically.
"""

from importlib import import_module
from pathlib import Path
from typing import List

from langchain_core.documents import Document


CHROMA_DB_PATH = Path("chroma_db")


vector_store_module = import_module(
    "05_create_chroma_store"
)

create_vector_store = (
    vector_store_module.create_vector_store
)

load_existing_vector_store = (
    vector_store_module.load_existing_vector_store
)


_vector_store = None


def get_vector_store():
    """
    Load ChromaDB or create it automatically
    when it does not exist.
    """

    global _vector_store

    if _vector_store is not None:
        return _vector_store

    if CHROMA_DB_PATH.exists():
        print("Loading existing Chroma database...")

        _vector_store = (
            load_existing_vector_store()
        )

    else:
        print(
            "Chroma database not found. "
            "Creating it automatically..."
        )

        _vector_store = create_vector_store(
            delete_existing=False
        )

    return _vector_store


def retrieve_context(
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Retrieve relevant document chunks.

    Args:
        query:
            User question.

        k:
            Number of chunks to retrieve.

    Returns:
        List of relevant LangChain documents.
    """

    if not query or not query.strip():
        raise ValueError(
            "The query cannot be empty."
        )

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query=query.strip(),
        k=k,
    )

    return results


def display_results(
    documents: List[Document]
):
    """
    Display retrieved documents in terminal.
    """

    print("=" * 60)
    print(
        f"Retrieved Results: {len(documents)}"
    )
    print("=" * 60)

    for index, document in enumerate(
        documents,
        start=1
    ):
        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page_number = document.metadata.get(
            "page_number"
        )

        if page_number is None:
            page = document.metadata.get(
                "page"
            )

            if page is not None:
                page_number = int(page) + 1
            else:
                page_number = "Unknown"

        chunk_index = document.metadata.get(
            "chunk_index",
            "Unknown"
        )

        print(f"\nResult {index}")
        print("-" * 60)
        print(document.page_content)

        print("\nMetadata")
        print("-" * 60)
        print(f"Source: {source}")
        print(f"Page: {page_number}")
        print(f"Chunk Index: {chunk_index}")

        print("=" * 60)


def main():
    try:
        query = input(
            "Ask a question: "
        ).strip()

        documents = retrieve_context(
            query=query,
            k=4
        )

        display_results(
            documents
        )

    except Exception as error:
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(type(error).__name__)
        print(error)


if __name__ == "__main__":
    main()
