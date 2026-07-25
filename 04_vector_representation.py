"""
04_vector_representation.py

Load the embedding model and test vector creation.
"""

from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    """
    Create and return the HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings:
            The embedding model used for converting
            text into numerical vectors.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embedding_model


def main():
    """
    Test the embedding model.
    """

    try:
        print("=" * 50)
        print("Loading Embedding Model...")
        print("=" * 50)

        embedding_model = get_embedding_model()

        test_text = (
            "Data analytics helps organizations "
            "make better decisions."
        )

        vector = embedding_model.embed_query(
            test_text
        )

        print(
            "Embedding Model Loaded Successfully"
        )
        print("-" * 50)

        print("Model Name:")
        print(MODEL_NAME)

        print("-" * 50)

        print(
            f"Vector Dimensions: {len(vector)}"
        )

        print("-" * 50)

        print("First 10 vector values:")

        print(vector[:10])

        print("=" * 50)

    except Exception as error:
        print("=" * 50)
        print("ERROR")
        print("=" * 50)
        print(error)


if __name__ == "__main__":
    main()