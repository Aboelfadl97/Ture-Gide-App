"""
05_create_chroma_store.py

Create and persist the Chroma vector database.
"""

from pathlib import Path
from importlib import import_module
import shutil

from langchain_chroma import Chroma


CHROMA_DB_PATH = Path("chroma_db")
COLLECTION_NAME = "digital_notes"


def create_vector_store(
    rebuild: bool = True
):
    """
    Create a Chroma vector store from the PDF documents.

    Args:
        rebuild:
            If True, delete the old database first.

    Returns:
        Chroma:
            The created vector store.
    """

    # Import files dynamically because
    # their names start with numbers.
    documents_module = import_module(
        "01_documents"
    )

    preprocessing_module = import_module(
        "02_preprocessing"
    )

    chunking_module = import_module(
        "03_chunking"
    )

    vector_module = import_module(
        "04_vector_representation"
    )

    load_documents = (
        documents_module.load_documents
    )

    preprocess_documents = (
        preprocessing_module
        .preprocess_documents
    )

    chunk_documents = (
        chunking_module.chunk_documents
    )

    get_embedding_model = (
        vector_module.get_embedding_model
    )

    # Delete the old vector store
    # to avoid duplicate chunks.
    if rebuild and CHROMA_DB_PATH.exists():
        print(
            "Deleting old Chroma database..."
        )

        shutil.rmtree(
            CHROMA_DB_PATH
        )

    print("=" * 50)
    print("1. Loading PDF documents...")
    print("=" * 50)

    documents = load_documents()

    print(
        f"Loaded PDF Pages: {len(documents)}"
    )

    print("=" * 50)
    print("2. Preprocessing documents...")
    print("=" * 50)

    cleaned_documents = (
        preprocess_documents(
            documents
        )
    )

    print(
        f"Cleaned Pages: "
        f"{len(cleaned_documents)}"
    )

    print("=" * 50)
    print("3. Creating chunks...")
    print("=" * 50)

    chunks = chunk_documents(
        cleaned_documents
    )

    if not chunks:
        raise ValueError(
            "No chunks were created. "
            "Please check your PDF document."
        )

    print(
        f"Created Chunks: {len(chunks)}"
    )

    print("=" * 50)
    print("4. Loading embedding model...")
    print("=" * 50)

    embedding_model = (
        get_embedding_model()
    )

    print(
        "Embedding model loaded."
    )

    print("=" * 50)
    print("5. Creating Chroma database...")
    print("=" * 50)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(
            CHROMA_DB_PATH
        ),
        collection_name=COLLECTION_NAME,
    )

    print("=" * 50)
    print(
        "Vector Database Created Successfully"
    )
    print("=" * 50)

    print(
        f"PDF Pages : {len(documents)}"
    )

    print(
        f"Clean Pages: "
        f"{len(cleaned_documents)}"
    )

    print(
        f"Chunks    : {len(chunks)}"
    )

    print(
        f"Database  : {CHROMA_DB_PATH}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print("=" * 50)

    return vector_store


def load_existing_vector_store():
    """
    Load the existing Chroma vector database.

    Returns:
        Chroma:
            The existing vector store.
    """

    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(
            "The Chroma database does not exist. "
            "Run 05_create_chroma_store.py first."
        )

    vector_module = import_module(
        "04_vector_representation"
    )

    get_embedding_model = (
        vector_module.get_embedding_model
    )

    embedding_model = (
        get_embedding_model()
    )

    vector_store = Chroma(
        persist_directory=str(
            CHROMA_DB_PATH
        ),
        embedding_function=(
            embedding_model
        ),
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def main():
    try:
        create_vector_store(
            rebuild=True
        )

    except Exception as error:
        print("=" * 50)
        print("ERROR")
        print("=" * 50)
        print(type(error).__name__)
        print(error)


if __name__ == "__main__":
    main()