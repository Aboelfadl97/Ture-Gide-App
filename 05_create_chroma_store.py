"""
05_create_chroma_store.py

Create and load the Chroma vector database.
"""

import shutil
from importlib import import_module
from pathlib import Path

from langchain_chroma import Chroma


CHROMA_DB_PATH = Path("chroma_db")
COLLECTION_NAME = "digital_notes"


documents_module = import_module("01_documents")
preprocessing_module = import_module("02_preprocessing")
chunking_module = import_module("03_chunking")
embedding_module = import_module("04_vector_representation")

load_documents = documents_module.load_documents
preprocess_documents = preprocessing_module.preprocess_documents
create_chunks = chunking_module.create_chunks
get_embedding_model = embedding_module.get_embedding_model


def create_vector_store(
    delete_existing: bool = False
) -> Chroma:
    """
    Create the Chroma database from the PDF documents.

    Args:
        delete_existing:
            Delete the old database before rebuilding.

    Returns:
        The created Chroma vector store.
    """

    if delete_existing and CHROMA_DB_PATH.exists():
        print("Deleting old Chroma database...")
        shutil.rmtree(CHROMA_DB_PATH)

    print("=" * 50)
    print("1. Loading PDF documents...")
    print("=" * 50)

    documents = load_documents()

    if not documents:
        raise ValueError(
            "No PDF documents were found "
            "inside the documents folder."
        )

    print("=" * 50)
    print("2. Preprocessing documents...")
    print("=" * 50)

    cleaned_documents = preprocess_documents(
        documents
    )

    print(f"Cleaned Pages: {len(cleaned_documents)}")

    print("=" * 50)
    print("3. Creating chunks...")
    print("=" * 50)

    chunks = create_chunks(
        cleaned_documents
    )

    if not chunks:
        raise ValueError(
            "No chunks were created from the PDF."
        )

    print(f"Created Chunks: {len(chunks)}")

    print("=" * 50)
    print("4. Loading embedding model...")
    print("=" * 50)

    embedding_model = get_embedding_model()

    print("Embedding model loaded.")

    print("=" * 50)
    print("5. Creating Chroma database...")
    print("=" * 50)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DB_PATH),
        collection_name=COLLECTION_NAME,
    )

    print("=" * 50)
    print("Vector Database Created Successfully")
    print("=" * 50)
    print(f"PDF Pages : {len(documents)}")
    print(f"Clean Pages: {len(cleaned_documents)}")
    print(f"Chunks    : {len(chunks)}")
    print(f"Database  : {CHROMA_DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print("=" * 50)

    return vector_store


def load_existing_vector_store() -> Chroma:
    """
    Load an existing Chroma database.
    """

    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(
            "The Chroma database does not exist."
        )

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )


def main():
    create_vector_store(
        delete_existing=True
    )


if __name__ == "__main__":
    main()
