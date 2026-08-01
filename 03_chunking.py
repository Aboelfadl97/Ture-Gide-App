"""الملف ده بقسم فيه النص لأجزاء صغيرة علشان البحث يبقى أدق."""

# الملف ده فيه تقسيم الصفحات لأجزاء صغيرة قابلة للبحث.
# حجم الجزء والتداخل قابلين للتغيير علشان نقدر نقارن بينهم في التجارب.

# هنا بجيب طريقة تسمحلي أحمّل ملفات بايثون أسماؤها بتبدأ بأرقام.
from importlib import import_module

# هنا بحدد شكل القوائم علشان بايثون يبقى فاهم نوع البيانات.
from typing import List

# هنا بجيب شكل المستند اللي بيحفظ النص ومعاه بيانات الصفحة.
from langchain_core.documents import Document

# هنا بجيب أداة تقسيم النص بشكل تدريجي مع احترام الفواصل الطبيعية.
from langchain_text_splitters import RecursiveCharacterTextSplitter


# هنا بحدد الحجم الافتراضي لكل جزء علشان ده الإعداد اللي طلع أحسن في التجارب.
DEFAULT_CHUNK_SIZE = 300

# هنا بحدد مقدار التداخل بين كل جزئين علشان آخر الكلام يفضل واصل بأول الجزء اللي بعده.
DEFAULT_CHUNK_OVERLAP = 50


preprocessing_module = import_module(
    "02_preprocessing"
)

preprocess_documents = (
    preprocessing_module.preprocess_documents
)


# الفانكشن دي بتقسم الصفحات لأجزاء صغيرة مع تداخل وتحفظ بيانات كل جزء.
def create_chunks(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """بقسم الصفحات لأجزاء صغيرة مع تداخل بينهم."""

    if not documents:
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller "
            "than chunk_size."
        )

    # هنا بعمل أداة التقسيم بالحجم والتداخل وترتيب الفواصل المفضل.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    # هنا النص بيتقسم فعليًا، وكل جزء بياخد بيانات صفحته.
    chunks = text_splitter.split_documents(
        documents
    )

    # هنا بدي لكل جزء رقم وأحفظ الإعدادات اللي اتعمل بيها.
    for chunk_index, chunk in enumerate(chunks):

        # هنا بحفظ رقم الجزء.
        chunk.metadata["chunk_index"] = (
            chunk_index
        )

        # هنا بسجل حجم الجزء.
        chunk.metadata["chunk_size"] = (
            chunk_size
        )

        # هنا بسجل مقدار التداخل.
        chunk.metadata["chunk_overlap"] = (
            chunk_overlap
        )

        # هنا بجيب رقم الصفحة اللي جه من الملف.
        original_page = chunk.metadata.get(
            "page"
        )

        if original_page is not None:
            chunk.metadata["page_number"] = (
                int(original_page) + 1
            )
        else:
            chunk.metadata["page_number"] = (
                "Unknown"
            )

    return chunks


# الفانكشن دي بتشغل اختبار الملف لما أشغله مباشرة من سطر الأوامر.
def main():
    """دي تجربة سريعة للملف لما أشغله لوحده."""

    try:

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

        chunks = create_chunks(
            cleaned_documents,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )

        print("=" * 60)

        print(
            f"Chunk Size: "
            f"{DEFAULT_CHUNK_SIZE}"
        )

        print(
            f"Chunk Overlap: "
            f"{DEFAULT_CHUNK_OVERLAP}"
        )

        print(
            f"Created Chunks: "
            f"{len(chunks)}"
        )

        print("=" * 60)

        for index, chunk in enumerate(
            chunks[:3],
            start=1
        ):

            print(
                f"\nChunk {index}"
            )

            print("-" * 60)

            print(
                chunk.page_content[:500]
            )

            print("\nMetadata:")

            print(
                chunk.metadata
            )

    except Exception as error:

        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(
            type(error).__name__
        )

        print(error)


if __name__ == "__main__":
    main()
