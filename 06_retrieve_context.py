"""الملف ده بجيب منه أنسب أجزاء للسؤال."""

# الملف ده بجيب منه أنسب أجزاء للسؤال.
# فيه بحث بالمعنى، وبالكلمات، ودمج بينهم.

# هنا بجيب طريقة تسمحلي أحمّل ملفات بايثون أسماؤها بتبدأ بأرقام.
from importlib import import_module

# هنا بجيب أداة التعامل مع مسارات الملفات والمجلدات.
from pathlib import Path

# هنا بجيب مكتبة الحسابات علشان أرتب درجات البحث.
import numpy as np

# هنا بجيب خوارزمية البحث بالكلمات وحساب أهميتها داخل المستندات.
from rank_bm25 import BM25Okapi


# هنا بحدد مكان حفظ قاعدة البيانات على الجهاز.
CHROMA_DB_PATH = Path("chroma_db")


vector_module = import_module(
    "05_create_chroma_store"
)

documents_module = import_module(
    "01_documents"
)

preprocessing_module = import_module(
    "02_preprocessing"
)

chunking_module = import_module(
    "03_chunking"
)


create_vector_store = (
    vector_module.create_vector_store
)

load_existing_vector_store = (
    vector_module.load_existing_vector_store
)

load_documents = (
    documents_module.load_documents
)

preprocess_documents = (
    preprocessing_module.preprocess_documents
)

tokenize_for_bm25 = (
    preprocessing_module.tokenize_for_bm25
)

create_chunks = (
    chunking_module.create_chunks
)


# هنا بحدد الحجم الافتراضي لكل جزء علشان ده الإعداد اللي طلع أحسن في التجارب.
DEFAULT_CHUNK_SIZE = 300

# هنا بحدد مقدار التداخل بين كل جزئين علشان آخر الكلام يفضل واصل بأول الجزء اللي بعده.
DEFAULT_CHUNK_OVERLAP = 50


_vector_store = None
_chunks = None
_bm25 = None

_current_chunk_size = None
_current_chunk_overlap = None


# الفانكشن دي بتمسح الحاجات المحفوظة في الذاكرة بين التجارب علشان كل تجربة تبدأ صح.
def reset_retrieval_cache():
    """بفضي الحاجات المحفوظة في الذاكرة قبل كل تجربة جديدة."""

    global _vector_store
    global _chunks
    global _bm25
    global _current_chunk_size
    global _current_chunk_overlap

    _vector_store = None
    _chunks = None
    _bm25 = None

    _current_chunk_size = None
    _current_chunk_overlap = None


# الفانكشن دي بتقرأ المستندات وتنضفها وتقسمها حسب الإعداد المطلوب.
def get_chunks(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    """بجهز أجزاء النص حسب الحجم والتداخل المطلوبين."""

    global _chunks
    global _current_chunk_size
    global _current_chunk_overlap

    configuration_changed = (
        _current_chunk_size != chunk_size
        or
        _current_chunk_overlap != chunk_overlap
    )

    if (
        _chunks is None
        or configuration_changed
    ):

        documents = load_documents()

        cleaned_documents = (
            preprocess_documents(documents)
        )

        _chunks = create_chunks(
            cleaned_documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        _current_chunk_size = chunk_size
        _current_chunk_overlap = chunk_overlap

    return _chunks


# الفانكشن دي بتعمل فهرس البحث بالكلمات على الأجزاء الحالية.
def get_bm25(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    """ببني فهرس البحث بالكلمات على الأجزاء الحالية."""

    global _bm25

    chunks = get_chunks(
        chunk_size,
        chunk_overlap
    )

    if _bm25 is None:

        tokenized_chunks = [
            tokenize_for_bm25(
                chunk.page_content
            )
            for chunk in chunks
        ]

        _bm25 = BM25Okapi(
            tokenized_chunks
        )

    return _bm25


# الفانكشن دي بتفتح قاعدة البيانات الحالية وبتسيبها في الذاكرة لتقليل وقت التحميل.
def get_vector_store():
    """بفتح قاعدة البيانات الحالية وأحتفظ بيها في الذاكرة."""

    global _vector_store

    if _vector_store is None:

        if not CHROMA_DB_PATH.exists():

            raise FileNotFoundError(
                "Chroma database not found."
            )

        _vector_store = (
            load_existing_vector_store()
        )

    return _vector_store


# الفانكشن دي بترجع أقرب أجزاء في المعنى للسؤال باستخدام المتجهات.
def semantic_retrieve(
    query,
    k=4
):
    """بجيب النتائج الأقرب في المعنى للسؤال."""

    vector_store = get_vector_store()

    return vector_store.similarity_search(
        query=query,
        k=k
    )


# الفانكشن دي بترجع الأجزاء اللي كلماتها أقرب لكلمات السؤال.
def bm25_retrieve(
    query,
    k=4,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    """بجيب النتائج اللي كلماتها أقرب لكلمات السؤال."""

    chunks = get_chunks(
        chunk_size,
        chunk_overlap
    )

    bm25 = get_bm25(
        chunk_size,
        chunk_overlap
    )

    # هنا بحول السؤال لكلمات بنفس طريقة تجهيز المستندات.
    query_tokens = tokenize_for_bm25(
        query
    )

    # هنا بحسب درجة مطابقة كل جزء مع كلمات السؤال.
    scores = bm25.get_scores(
        query_tokens
    )

    # هنا برتب الدرجات من الأكبر للأصغر وأختار أفضل النتائج.
    top_indices = np.argsort(
        scores
    )[::-1][:k]

    results = []

    for index in top_indices:

        document = chunks[int(index)]

        document.metadata[
            "bm25_score"
        ] = float(
            scores[index]
        )

        results.append(document)

    return results


# الفانكشن دي بتدمج ترتيب البحث الدلالي والبحث بالكلمات في ترتيب واحد.
def hybrid_retrieve(
    query,
    k=4,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    alpha=0.5
):
    """بدمج ترتيب البحث بالمعنى والبحث بالكلمات."""

    # هنا بجيب عدد مرشحين أكبر قبل الدمج علشان البحث الهجين يبقى عنده اختيارات كفاية.
    candidate_k = max(
        k * 3,
        10
    )

    semantic_results = (
        semantic_retrieve(
            query,
            candidate_k
        )
    )

    bm25_results = (
        bm25_retrieve(
            query,
            candidate_k,
            chunk_size,
            chunk_overlap
        )
    )

    scores = {}
    documents = {}

    # هنا بثبت قيمة مستخدمة في دمج الترتيب وتقليل تأثير الفروق الكبيرة بين المراكز.
    rrf_constant = 60

    for rank, document in enumerate(
        semantic_results,
        start=1
    ):

        chunk_id = document.metadata.get(
            "chunk_index"
        )

        documents[chunk_id] = document

        score = (
            alpha
            / (rrf_constant + rank)
        )

        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + score
        )

    for rank, document in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = document.metadata.get(
            "chunk_index"
        )

        documents[chunk_id] = document

        score = (
            (1 - alpha)
            / (rrf_constant + rank)
        )

        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + score
        )

    # هنا برتب الأجزاء بعد جمع درجات الطريقتين.
    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    results = []

    for chunk_id in ranked_ids[:k]:

        document = documents[
            chunk_id
        ]

        document.metadata[
            "hybrid_score"
        ] = scores[chunk_id]

        results.append(document)

    return results


# دي الفانكشن الأساسية اللي بتختار طريقة الاسترجاع المطلوبة وترجع النتائج.
def retrieve_context(
    query,
    k=4,
    retriever_type="semantic",
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    """دي الدالة الأساسية اللي بتختار طريقة البحث المطلوبة."""

    if not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    retriever_type = (
        retriever_type.lower()
    )

    if retriever_type == "semantic":

        return semantic_retrieve(
            query,
            k
        )

    elif retriever_type == "bm25":

        return bm25_retrieve(
            query,
            k,
            chunk_size,
            chunk_overlap
        )

    elif retriever_type == "hybrid":

        return hybrid_retrieve(
            query,
            k,
            chunk_size,
            chunk_overlap
        )

    else:

        raise ValueError(
            "Retriever must be semantic, "
            "bm25, or hybrid."
        )


# الفانكشن دي بتشغل اختبار الملف لما أشغله مباشرة من سطر الأوامر.
def main():

    query = input(
        "Ask a question: "
    ).strip()

    for retriever in [
        "semantic",
        "bm25",
        "hybrid"
    ]:

        print("\n" + "=" * 60)
        print(retriever.upper())
        print("=" * 60)

        results = retrieve_context(
            query=query,
            k=4,
            retriever_type=retriever
        )

        for i, document in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {i}"
            )

            print(
                f"Page: "
                f"{document.metadata.get('page_number')}"
            )

            print(
                f"Chunk: "
                f"{document.metadata.get('chunk_index')}"
            )

            print(
                document.page_content[:300]
            )


if __name__ == "__main__":
    main()
