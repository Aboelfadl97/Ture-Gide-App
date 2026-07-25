"""
06_retrieve_context.py

Load the existing Chroma vector database
and retrieve the most relevant chunks.
"""

from pathlib import Path
from importlib import import_module
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma


CHROMA_DB_PATH = Path("chroma_db")
COLLECTION_NAME = "digital_notes"


def load_vector_store() -> Chroma:
    """
    Load the existing Chroma vector database.

    Returns:
        Chroma:
            The loaded vector store.
    """

    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(
            "The Chroma database does not exist. "
            "Run 05_create_chroma_store.py first."
        )

    # Dynamic import because the filename
    # starts with a number.
    vector_module = import_module(
        "04_vector_representation"
    )

    get_embedding_model = (
        vector_module.get_embedding_model
    )

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=str(
            CHROMA_DB_PATH
        ),
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def retrieve_context(
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks.

    Args:
        query:
            The user's question.

        k:
            Number of chunks to retrieve.

    Returns:
        List of relevant LangChain Documents.
    """

    if not query or not query.strip():
        raise ValueError(
            "The query cannot be empty."
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query=query.strip(),
        k=k
    )

    return results


def print_results(
    documents: List[Document]
) -> None:
    """
    Print retrieved chunks with metadata.
    """

    if not documents:
        print(
            "No relevant documents were found."
        )
        return

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

        # Fallback if page_number was not saved
        if page_number is None:
            original_page = document.metadata.get(
                "page"
            )

            if original_page is not None:
                page_number = int(
                    original_page
                ) + 1
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
        print(
            f"Chunk Index: {chunk_index}"
        )

        print("=" * 60)


def main():
    """
    Test context retrieval from the terminal.
    """

    try:
        question = input(
            "Ask a question: "
        ).strip()

        documents = retrieve_context(
            query=question,
            k=4
        )

        print_results(documents)

    except Exception as error:
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(type(error).__name__)
        print(error)


if __name__ == "__main__":
    main()