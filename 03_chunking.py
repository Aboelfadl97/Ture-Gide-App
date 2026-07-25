"""
03_chunking.py

Split cleaned PDF documents into smaller chunks.
"""

from importlib import import_module
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_documents(
    documents: List[Document]
) -> List[Document]:
    """
    Split cleaned documents into smaller chunks.

    Args:
        documents:
            List of cleaned LangChain Document objects.

    Returns:
        List of chunked Document objects.
    """

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ".",
            " ",
            ""
        ],
        length_function=len,
    )

    chunks = splitter.split_documents(
        documents
    )

    # Add chunk index to metadata
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

        # Add readable page number
        original_page = chunk.metadata.get(
            "page"
        )

        if original_page is not None:
            chunk.metadata["page_number"] = (
                int(original_page) + 1
            )

    return chunks


def main():
    """
    Load, preprocess, and test chunking.
    """

    try:
        documents_module = import_module(
            "01_documents"
        )

        preprocessing_module = import_module(
            "02_preprocessing"
        )

        load_documents = (
            documents_module.load_documents
        )

        preprocess_documents = (
            preprocessing_module
            .preprocess_documents
        )

        raw_documents = load_documents()

        cleaned_documents = (
            preprocess_documents(
                raw_documents
            )
        )

        chunks = chunk_documents(
            cleaned_documents
        )

        print("=" * 50)
        print(
            f"Loaded PDF Pages: "
            f"{len(raw_documents)}"
        )
        print(
            f"Cleaned Pages: "
            f"{len(cleaned_documents)}"
        )
        print(
            f"Created Chunks: "
            f"{len(chunks)}"
        )
        print("=" * 50)

        if chunks:
            first_chunk = chunks[0]

            print("First chunk preview:")
            print("-" * 50)
            print(
                first_chunk.page_content[:1000]
            )
            print("-" * 50)

            print(
                "Source:",
                first_chunk.metadata.get(
                    "source",
                    "Unknown"
                )
            )

            print(
                "PDF Page:",
                first_chunk.metadata.get(
                    "page_number",
                    "Unknown"
                )
            )

            print(
                "Chunk Index:",
                first_chunk.metadata.get(
                    "chunk_index",
                    "Unknown"
                )
            )

            print(
                "Chunk Length:",
                len(first_chunk.page_content)
            )

    except Exception as error:
        print("=" * 50)
        print("ERROR")
        print("=" * 50)
        print(error)


if __name__ == "__main__":
    main()