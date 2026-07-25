"""
01_documents.py

Load PDF documents from the documents folder.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


# Folder that contains the PDF files
DOCUMENTS_FOLDER = Path("documents")


def load_documents():
    """
    Load all PDF files from the documents folder.

    Returns:
        list: A list of LangChain Document objects.
              Each PDF page becomes one Document.
    """

    documents = []

    # Check that the documents folder exists
    if not DOCUMENTS_FOLDER.exists():
        raise FileNotFoundError(
            f"Folder '{DOCUMENTS_FOLDER}' does not exist."
        )

    # Find all PDF files
    pdf_files = list(DOCUMENTS_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found inside '{DOCUMENTS_FOLDER}'."
        )

    # Load every PDF file
    for file_path in pdf_files:
        print(f"Loading: {file_path}")

        loader = PyPDFLoader(str(file_path))

        loaded_pages = loader.load()

        documents.extend(loaded_pages)

    return documents


def main():
    try:
        docs = load_documents()

        print("=" * 50)
        print(f"Loaded PDF Pages: {len(docs)}")
        print("=" * 50)

        for doc in docs[:5]:
            print(
                f"Source: {doc.metadata.get('source', 'Unknown')}"
            )
            print(
                f"Page: {doc.metadata.get('page', 'Unknown')}"
            )
            print("-" * 50)

    except Exception as error:
        print("=" * 50)
        print("ERROR")
        print("=" * 50)
        print(error)


if __name__ == "__main__":
    main()