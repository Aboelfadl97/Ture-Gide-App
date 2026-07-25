"""
03_chunking.py

Split cleaned PDF documents into smaller chunks.
"""

from importlib import import_module
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


preprocessing_module = import_module(
    "02_preprocessing"
)

preprocess_documents = (
    preprocessing_module.preprocess_documents
)


def create_chunks(
    documents: List[Document]
) -> List[Document]:
    """
    Split documents into smaller overlapping chunks.

    Args:
        documents:
            Cleaned LangChain documents.

    Returns:
        List of chunked LangChain documents.
    """

    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(
        documents
    )

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index

        if "page_number" not in chunk.metadata:
            original_page = chunk.metadata.get("page")

            if original_page is not None:
                chunk.metadata["page_number"] = (
                    int(original_page) + 1
                )
            else:
                chunk.metadata["page_number"] = "Unknown"

    return chunks


def main():
    """
    Test document preprocessing and chunking.
    """

    try:
        documents = preprocess_documents()

        chunks = create_chunks(
            documents
        )

        print("=" * 50)
        print(f"Created Chunks: {len(chunks)}")
        print("=" * 50)

        for index, chunk in enumerate(
            chunks[:3],
            start=1
        ):
            print(f"\nChunk {index}")
            print("-" * 50)
            print(chunk.page_content[:500])
            print("\nMetadata:")
            print(chunk.metadata)

    except Exception as error:
        print("=" * 50)
        print("ERROR")
        print("=" * 50)
        print(type(error).__name__)
        print(error)


if __name__ == "__main__":
    main()
