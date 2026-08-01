"""الملف ده ببني فيه قاعدة البيانات اللي هتخزن أجزاء النص."""

# الملف ده ببني فيه قاعدة البيانات وبخزن الأجزاء جواها.
# هنا بجمع خطوات التحميل والتنظيف والتقسيم والتحويل لأرقام.

# هنا بجيب أداة حذف المجلدات القديمة وقت إعادة البناء.
import shutil

# هنا بجيب طريقة تسمحلي أحمّل ملفات بايثون أسماؤها بتبدأ بأرقام.
from importlib import import_module

# هنا بجيب أداة التعامل مع مسارات الملفات والمجلدات.
from pathlib import Path

# هنا بجيب قاعدة البيانات اللي هنخزن فيها الأجزاء.
from langchain_chroma import Chroma


# هنا بحدد مكان حفظ قاعدة البيانات على الجهاز.
CHROMA_DB_PATH = Path("chroma_db")

# هنا بكتب اسم المجموعة اللي هتتحفظ جوه قاعدة البيانات.
COLLECTION_NAME = "digital_notes"


documents_module = import_module(
    "01_documents"
)

preprocessing_module = import_module(
    "02_preprocessing"
)

chunking_module = import_module(
    "03_chunking"
)

embedding_module = import_module(
    "04_vector_representation"
)


load_documents = (
    documents_module.load_documents
)

preprocess_documents = (
    preprocessing_module.preprocess_documents
)

create_chunks = (
    chunking_module.create_chunks
)

get_embedding_model = (
    embedding_module.get_embedding_model
)

# هنا بحدد الحجم الافتراضي لكل جزء علشان ده الإعداد اللي طلع أحسن في التجارب.
DEFAULT_CHUNK_SIZE = (
    chunking_module.DEFAULT_CHUNK_SIZE
)

# هنا بحدد مقدار التداخل بين كل جزئين علشان آخر الكلام يفضل واصل بأول الجزء اللي بعده.
DEFAULT_CHUNK_OVERLAP = (
    chunking_module.DEFAULT_CHUNK_OVERLAP
)

# هنا بكتب اسم الموديل اللي هنستخدمه.
DEFAULT_MODEL_NAME = (
    embedding_module.DEFAULT_MODEL_NAME
)


# الفانكشن دي بتبني قاعدة البيانات من أول تحميل المستندات لحد حفظ الأجزاء.
def create_vector_store(
    delete_existing: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    model_name: str = DEFAULT_MODEL_NAME
) -> Chroma:
    """ببني قاعدة البيانات من أول تحميل الملف لحد تخزين الأجزاء."""

    if (
        delete_existing
        and CHROMA_DB_PATH.exists()
    ):
        print(
            "Deleting old Chroma database..."
        )

        shutil.rmtree(
            CHROMA_DB_PATH
        )

    print("=" * 60)
    print("1. Loading PDF documents...")
    print("=" * 60)

    documents = load_documents()

    if not documents:
        raise ValueError(
            "No PDF documents were found."
        )

    print("=" * 60)
    print("2. Preprocessing documents...")
    print("=" * 60)

    cleaned_documents = (
        preprocess_documents(documents)
    )

    print(
        f"Cleaned Pages: "
        f"{len(cleaned_documents)}"
    )

    print("=" * 60)
    print("3. Creating chunks...")
    print("=" * 60)

    chunks = create_chunks(
        cleaned_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    print(
        f"Chunk Size: {chunk_size}"
    )

    print(
        f"Chunk Overlap: {chunk_overlap}"
    )

    print(
        f"Created Chunks: {len(chunks)}"
    )

    print("=" * 60)
    print("4. Loading embedding model...")
    print("=" * 60)

    embedding_model = (
        get_embedding_model(
            model_name=model_name
        )
    )

    print(
        f"Embedding Model: {model_name}"
    )

    print("=" * 60)
    print("5. Creating Chroma database...")
    print("=" * 60)

    # هنا ببدأ أبني قاعدة البيانات من الأجزاء والموديل.
    vector_store = (

        # هنا كل جزء بيتحول لأرقام ويتخزن في قاعدة البيانات.
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(
                CHROMA_DB_PATH
            ),
            collection_name=COLLECTION_NAME,
        )
    )

    print("=" * 60)
    print(
        "Vector Database Created Successfully"
    )
    print("=" * 60)

    print(
        f"PDF Pages     : {len(documents)}"
    )

    print(
        f"Clean Pages   : {len(cleaned_documents)}"
    )

    print(
        f"Chunks        : {len(chunks)}"
    )

    print(
        f"Chunk Size    : {chunk_size}"
    )

    print(
        f"Chunk Overlap : {chunk_overlap}"
    )

    print(
        f"Embedding     : {model_name}"
    )

    print(
        f"Database      : {CHROMA_DB_PATH}"
    )

    print(
        f"Collection    : {COLLECTION_NAME}"
    )

    print("=" * 60)

    return vector_store


# الفانكشن دي بتفتح قاعدة بيانات اتبنت قبل كده بدل ما نعيد بناءها.
def load_existing_vector_store(
    model_name: str = DEFAULT_MODEL_NAME
) -> Chroma:
    """بفتح قاعدة البيانات الموجودة بدل ما أبنيها من جديد."""

    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(
            "The Chroma database does not exist."
        )

    embedding_model = (
        get_embedding_model(
            model_name=model_name
        )
    )

    return Chroma(
        persist_directory=str(
            CHROMA_DB_PATH
        ),
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )


# الفانكشن دي بتشغل اختبار الملف لما أشغله مباشرة من سطر الأوامر.
def main():
    """دي تجربة سريعة للملف لما أشغله لوحده."""

    # هنا ببني قاعدة بيانات خاصة بالإعداد الحالي لو مش موجودة.
    create_vector_store(
        delete_existing=True,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        model_name=DEFAULT_MODEL_NAME
    )


if __name__ == "__main__":
    main()
