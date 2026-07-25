"""
02_preprocessing.py

Clean and preprocess the loaded PDF documents.
"""

import re
from importlib import import_module
from typing import List

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Clean text extracted from PDF files.

    The function:
    - Removes null characters.
    - Removes unnecessary spaces.
    - Preserves paragraph breaks.
    - Removes excessive blank lines.
    """

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Remove soft hyphen characters
    text = text.replace("\u00ad", "")

    # Join words broken across lines using a hyphen
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # Normalize tabs and repeated spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove spaces around line breaks
    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    # Replace more than two line breaks with two
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def preprocess_documents(
    documents: List[Document]
) -> List[Document]:
    """
    Apply text cleaning to all loaded documents.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of cleaned Document objects.
    """

    cleaned_documents = []

    for document in documents:
        cleaned_text = clean_text(
            document.page_content
        )

        # Skip empty pages
        if not cleaned_text:
            continue

        cleaned_document = Document(
            page_content=cleaned_text,
            metadata=document.metadata.copy()
        )

        cleaned_documents.append(
            cleaned_document
        )

    return cleaned_documents


def main():
    """
    Load PDF documents and test preprocessing.
    """

    try:
        # Import 01_documents.py dynamically because
        # the filename starts with a number.
        documents_module = import_module(
            "01_documents"
        )

        load_documents = (
            documents_module.load_documents
        )

        documents = load_documents()

        cleaned_documents = (
            preprocess_documents(documents)
        )

        print("=" * 50)
        print(
            f"Original PDF Pages: "
            f"{len(documents)}"
        )
        print(
            f"Cleaned PDF Pages: "
            f"{len(cleaned_documents)}"
        )
        print("=" * 50)

        if cleaned_documents:
            first_document = cleaned_documents[0]

            print("First cleaned page preview:")
            print("-" * 50)
            print(
                first_document.page_content[:1000]
            )
            print("-" * 50)

            print(
                "Source:",
                first_document.metadata.get(
                    "source",
                    "Unknown"
                )
            )

            print(
                "Page:",
                first_document.metadata.get(
                    "page",
                    "Unknown"
                )
            )

    except Exception as error:
        print("=" * 50)
        print("ERROR")
        print("=" * 50)
        print(error)


if __name__ == "__main__":
    main()