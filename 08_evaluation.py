"""الملف ده بقيس فيه جودة الاسترجاع بالأرقام."""

# الملف ده بقيس فيه جودة البحث باستخدام أسئلة معروفة إجاباتها.
# هنا بحسب الأربع مقاييس اللي استخدمناهم في المقارنة.

# هنا بجيب أدوات التعامل مع ملفات الجداول النصية المستخدمة في التقييم.
import csv

# هنا بجيب طريقة تسمحلي أحمّل ملفات بايثون أسماؤها بتبدأ بأرقام.
from importlib import import_module


retrieval_module = import_module(
    "06_retrieve_context"
)

retrieve_context = (
    retrieval_module.retrieve_context
)


# هنا بحدد ملف الأسئلة والصفحات الصح.
GROUND_TRUTH_FILE = "ground_truth.csv"

# هنا بحط طرق البحث اللي هقارن بينها.
RETRIEVERS = [
    "semantic",
    "bm25",
    "hybrid"
]

# هنا بحط أعداد النتائج اللي هجربها.
TOP_K_VALUES = [
    2,
    4,
    6
]


# الفانكشن دي بتقرأ أسئلة التقييم والإجابات والصفحات الصح من الملف.
def load_ground_truth():
    """بقرأ أسئلة التقييم والإجابات والصفحات الصح."""

    dataset = []

    # هنا بفتح ملف التقييم.
    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:

        # هنا بقرأ كل سؤال ومعاه باقي بياناته.
        reader = csv.DictReader(file)

        for row in reader:

            # هنا بحول أرقام الصفحات لقائمة أرقام فعلية.
            relevant_pages = [
                int(page.strip())
                for page
                in row["relevant_pages"].split("|")
            ]

            dataset.append({
                "question": row["question"],
                "expected_answer": row[
                    "expected_answer"
                ],
                "relevant_pages": relevant_pages
            })

    return dataset


# الفانكشن دي بتطلع أرقام الصفحات اللي رجعتها طريقة الاسترجاع.
def get_retrieved_pages(documents):
    """بطلع أرقام الصفحات اللي رجعها البحث."""

    pages = []

    # هنا بلف على النتائج واحدة واحدة.
    for document in documents:

        page = document.metadata.get(
            "page_number"
        )

        if page is not None:

            page = int(page)

            if page not in pages:
                pages.append(page)

    return pages


# الفانكشن دي بتحسب مقاييس التقييم بمقارنة الصفحات المسترجعة بالصفحات الصح.
def calculate_metrics(
    retrieved_pages,
    relevant_pages,
    k
):
    """بحسب مقاييس التقييم من الصفحات الصح والصفحات المسترجعة."""

    retrieved_pages = retrieved_pages[:k]

    relevant_set = set(relevant_pages)

    # هنا بشوف أنهي صفحات من اللي رجعت كانت صح.
    hits = [
        page
        for page in retrieved_pages
        if page in relevant_set
    ]

    # هنا بشوف هل ظهر جواب صح ضمن النتائج ولا لأ.
    hit_rate = (
        1.0
        if len(hits) > 0
        else 0.0
    )

    # هنا بحسب قد إيه من الصفحات الصح رجعناها.
    recall = (
        len(set(hits))
        / len(relevant_set)
        if relevant_set
        else 0.0
    )

    # هنا بحسب قد إيه من النتائج اللي رجعت كانت صح.
    precision = (
        len(hits) / k
        if k > 0
        else 0.0
    )

    # هنا بخلي قيمة ترتيب أول نتيجة صح تبدأ من صفر قبل البحث عن أول نتيجة صحيحة.
    reciprocal_rank = 0.0

    # هنا بشوف أول صفحة صح ظهرت في ترتيب كام.
    for rank, page in enumerate(
        retrieved_pages,
        start=1
    ):

        if page in relevant_set:

            reciprocal_rank = (
                1.0 / rank
            )

            break

    return {
        "hit_rate": hit_rate,
        "recall": recall,
        "precision": precision,
        "mrr": reciprocal_rank
    }


# الفانكشن دي بتجرب طريقة بحث واحدة على كل أسئلة التقييم وتطلع المتوسطات.
def evaluate_retriever(
    dataset,
    retriever_type,
    k
):
    """بجرب طريقة بحث واحدة على كل أسئلة التقييم."""

    metrics_list = []

    for index, item in enumerate(
        dataset,
        start=1
    ):

        question = item["question"]

        relevant_pages = item[
            "relevant_pages"
        ]

        # هنا بجيب أنسب أجزاء للسؤال الأول.
        documents = retrieve_context(
            query=question,
            k=k,
            retriever_type=retriever_type
        )

        retrieved_pages = (
            get_retrieved_pages(
                documents
            )
        )

        metrics = calculate_metrics(
            retrieved_pages,
            relevant_pages,
            k
        )

        metrics_list.append(metrics)

        print(
            f"[{index}/{len(dataset)}] "
            f"{retriever_type.upper()} "
            f"K={k}"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Relevant: {relevant_pages}"
        )

        print(
            f"Retrieved: {retrieved_pages}"
        )

        print("-" * 60)

    total = len(metrics_list)

    return {
        "retriever": retriever_type,
        "top_k": k,

        "hit_rate": sum(
            item["hit_rate"]
            for item in metrics_list
        ) / total,

        "recall": sum(
            item["recall"]
            for item in metrics_list
        ) / total,

        "precision": sum(
            item["precision"]
            for item in metrics_list
        ) / total,

        "mrr": sum(
            item["mrr"]
            for item in metrics_list
        ) / total
    }


# الفانكشن دي بتشغل اختبار الملف لما أشغله مباشرة من سطر الأوامر.
def main():

    dataset = load_ground_truth()

    print("=" * 70)

    print(
        f"Ground Truth Questions: "
        f"{len(dataset)}"
    )

    print("=" * 70)

    final_results = []

    for k in TOP_K_VALUES:

        for retriever in RETRIEVERS:

            print(
                f"\nEvaluating "
                f"{retriever.upper()} "
                f"with Top-K = {k}"
            )

            print("=" * 70)

            result = evaluate_retriever(
                dataset=dataset,
                retriever_type=retriever,
                k=k
            )

            final_results.append(result)

    print("\n")
    print("=" * 86)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 86)

    print(
        f"{'Retriever':<12}"
        f"{'K':<6}"
        f"{'Hit Rate':<14}"
        f"{'Recall':<14}"
        f"{'Precision':<14}"
        f"{'MRR':<14}"
    )

    print("-" * 86)

    for result in final_results:

        print(
            f"{result['retriever']:<12}"
            f"{result['top_k']:<6}"
            f"{result['hit_rate']:<14.3f}"
            f"{result['recall']:<14.3f}"
            f"{result['precision']:<14.3f}"
            f"{result['mrr']:<14.3f}"
        )


if __name__ == "__main__":
    main()
