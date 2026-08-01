"""الملف ده بجرب فيه أكتر من حجم وتقسيمة وأقارن النتائج."""

# الملف ده بجرب فيه أكتر من حجم وتقسيمة.
# كل إعداد بيتقاس بنفس الأسئلة ونفس طرق الاسترجاع علشان المقارنة تبقى عادلة.

# هنا بجيب أدوات التعامل مع ملفات الجداول النصية المستخدمة في التقييم.
import csv

# هنا بجيب طريقة تسمحلي أحمّل ملفات بايثون أسماؤها بتبدأ بأرقام.
from importlib import import_module

# هنا بجيب أداة التعامل مع مسارات الملفات والمجلدات.
from pathlib import Path


evaluation_module = import_module(
    "08_evaluation"
)

retrieval_module = import_module(
    "06_retrieve_context"
)

vector_module = import_module(
    "05_create_chroma_store"
)


load_ground_truth = (
    evaluation_module.load_ground_truth
)

calculate_metrics = (
    evaluation_module.calculate_metrics
)

get_retrieved_pages = (
    evaluation_module.get_retrieved_pages
)

retrieve_context = (
    retrieval_module.retrieve_context
)

reset_retrieval_cache = (
    retrieval_module.reset_retrieval_cache
)

create_vector_store = (
    vector_module.create_vector_store
)


# هنا بحط كل أحجام التقسيم والتداخل اللي هجربها.
CHUNK_CONFIGURATIONS = [
    (300, 50),
    (500, 100),
    (700, 150),
    (1000, 200),
]

# هنا بحط طرق البحث اللي هقارن بينها.
RETRIEVERS = [
    "semantic",
    "bm25",
    "hybrid",
]

# هنا بخلي عدد النتائج ثابت في كل تجربة.
TOP_K = 2

# هنا بحدد اسم ملف النتائج.
OUTPUT_FILE = (
    "chunk_experiment_results.csv"
)

# هنا بحدد مكان قواعد بيانات التجارب.
EXPERIMENT_FOLDER = Path(
    "chroma_experiments"
)


# الفانكشن دي بتعمل لكل إعداد قاعدة بيانات لوحدها علشان الملفات ما تتقفلش على بعض.
def set_chroma_path(
    chunk_size,
    chunk_overlap
):
    """بعمل لكل تجربة مجلد قاعدة بيانات لوحده."""

    experiment_path = (
        EXPERIMENT_FOLDER
        / (
            f"chroma_"
            f"{chunk_size}_"
            f"{chunk_overlap}"
        )
    )

    vector_module.CHROMA_DB_PATH = (
        experiment_path
    )

    retrieval_module.CHROMA_DB_PATH = (
        experiment_path
    )

    return experiment_path


# الفانكشن دي بتقيم إعداد واحد كامل من حجم جزء وتداخل وطريقة استرجاع.
def evaluate_configuration(
    dataset,
    retriever,
    chunk_size,
    chunk_overlap
):

    metrics_list = []

    for item in dataset:

        # هنا بجيب أنسب أجزاء للسؤال الأول.
        documents = retrieve_context(
            query=item["question"],
            k=TOP_K,
            retriever_type=retriever,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        retrieved_pages = (
            get_retrieved_pages(
                documents
            )
        )

        metrics = calculate_metrics(
            retrieved_pages,
            item["relevant_pages"],
            TOP_K
        )

        metrics_list.append(
            metrics
        )

    total = len(
        metrics_list
    )

    return {
        "chunk_size":
            chunk_size,

        "chunk_overlap":
            chunk_overlap,

        "retriever":
            retriever,

        "top_k":
            TOP_K,

        "hit_rate":
            sum(
                x["hit_rate"]
                for x in metrics_list
            ) / total,

        "recall":
            sum(
                x["recall"]
                for x in metrics_list
            ) / total,

        "precision":
            sum(
                x["precision"]
                for x in metrics_list
            ) / total,

        "mrr":
            sum(
                x["mrr"]
                for x in metrics_list
            ) / total,
    }


# الفانكشن دي بتشغل اختبار الملف لما أشغله مباشرة من سطر الأوامر.
def main():

    dataset = (
        load_ground_truth()
    )

    if not dataset:
        raise ValueError(
            "Ground Truth is empty."
        )

    EXPERIMENT_FOLDER.mkdir(
        exist_ok=True
    )

    # هنا بعمل قائمة هتجمع نتائج كل التجارب.
    all_results = []

    # هنا بلف على كل إعداد من إعدادات التقسيم.
    for (
        chunk_size,
        chunk_overlap
    ) in CHUNK_CONFIGURATIONS:

        print("\n")
        print("=" * 70)

        print(
            f"CHUNK SIZE = "
            f"{chunk_size} | "
            f"OVERLAP = "
            f"{chunk_overlap}"
        )

        print("=" * 70)

        reset_retrieval_cache()

        experiment_path = (
            set_chroma_path(
                chunk_size,
                chunk_overlap
            )
        )

        print(
            f"Database: "
            f"{experiment_path}"
        )

        if not experiment_path.exists():

            # هنا ببني قاعدة بيانات خاصة بالإعداد الحالي لو مش موجودة.
            create_vector_store(
                delete_existing=False,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        else:

            print(
                "Existing experiment "
                "database found."
            )

        reset_retrieval_cache()

        for retriever in RETRIEVERS:

            print(
                f"\nEvaluating: "
                f"{retriever.upper()}"
            )

            result = (
                evaluate_configuration(
                    dataset=dataset,
                    retriever=retriever,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            )

            all_results.append(
                result
            )

            print(
                f"Hit Rate : "
                f"{result['hit_rate']:.3f}"
            )

            print(
                f"Recall   : "
                f"{result['recall']:.3f}"
            )

            print(
                f"Precision: "
                f"{result['precision']:.3f}"
            )

            print(
                f"MRR      : "
                f"{result['mrr']:.3f}"
            )

        reset_retrieval_cache()

    # هنا بفتح ملف التقييم.
    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "chunk_size",
            "chunk_overlap",
            "retriever",
            "top_k",
            "hit_rate",
            "recall",
            "precision",
            "mrr",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    print("\n")
    print("=" * 90)

    print(
        "FINAL CHUNKING EXPERIMENT RESULTS"
    )

    print("=" * 90)

    print(
        f"{'Chunk':<10}"
        f"{'Overlap':<10}"
        f"{'Retriever':<12}"
        f"{'Hit':<10}"
        f"{'Recall':<10}"
        f"{'Precision':<12}"
        f"{'MRR':<10}"
    )

    print("-" * 90)

    for result in all_results:

        print(
            f"{result['chunk_size']:<10}"
            f"{result['chunk_overlap']:<10}"
            f"{result['retriever']:<12}"
            f"{result['hit_rate']:<10.3f}"
            f"{result['recall']:<10.3f}"
            f"{result['precision']:<12.3f}"
            f"{result['mrr']:<10.3f}"
        )

    print("=" * 90)

    # هنا باختار أفضل إعداد حسب ترتيب المقاييس المحدد في الكود.
    best = max(
        all_results,
        key=lambda x: (
            x["mrr"],
            x["hit_rate"],
            x["precision"]
        )
    )

    print("\nBEST CONFIGURATION")

    print(
        f"Chunk Size : "
        f"{best['chunk_size']}"
    )

    print(
        f"Overlap    : "
        f"{best['chunk_overlap']}"
    )

    print(
        f"Retriever  : "
        f"{best['retriever']}"
    )

    print(
        f"Top-K      : "
        f"{best['top_k']}"
    )

    print(
        f"Hit Rate   : "
        f"{best['hit_rate']:.3f}"
    )

    print(
        f"MRR        : "
        f"{best['mrr']:.3f}"
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
