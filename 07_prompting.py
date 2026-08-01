"""الملف ده بجهز فيه السؤال والسياق وببعتهم للنموذج."""

# الملف ده بجهز فيه الكلام اللي هيتبعت للموديل.
# كمان بيرجع الإجابة ومعاها المصادر من غير اختراع معلومات.

# هنا بجيب أدوات التعامل مع متغيرات البيئة ومفتاح الخدمة.
import os

# هنا بجيب طريقة تسمحلي أحمّل ملفات بايثون أسماؤها بتبدأ بأرقام.
from importlib import import_module
from typing import List, Tuple

# هنا بجيب أداة قراءة القيم السرية من ملف البيئة المحلي.
from dotenv import load_dotenv

# هنا بجيب الأداة اللي هتكلم الخدمة اللي بتشغل النموذج اللغوي.
from groq import Groq

# هنا بجيب شكل المستند اللي بيحفظ النص ومعاه بيانات الصفحة.
from langchain_core.documents import Document


# هنا بقرأ القيم السرية من الملف المحلي.
load_dotenv()


# هنا باخد المفتاح من بره الكود علشان مايبقاش مكشوف.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# هنا بكتب اسم الموديل اللي هيطلع الإجابة.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


retrieval_module = import_module(
    "06_retrieve_context"
)

retrieve_context = (
    retrieval_module.retrieve_context
)


# هنا بكتب القواعد اللي تخلي النموذج يجاوب من الكلام اللي رجعناه بس.
PROMPT_TEMPLATE = """
You are True Guide, a Data Analytics study assistant.

Answer the user's question using ONLY the provided context
from the Digital Notes PDF.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the context, say:
   "I could not find this information in the Digital Notes."
4. Explain the answer clearly and simply.
5. Keep the answer focused and organized.
6. Do not invent page numbers or sources.

Retrieved context:
{context}

Question:
{question}

Answer:
"""


# الفانكشن دي بتحول رقم الصفحة للشكل اللي المستخدم يفهمه وتتعامل مع القيم الناقصة.
def get_page_number(
    document: Document
):
    """بطلع رقم الصفحة بالشكل اللي المستخدم يشوفه."""

    page_number = document.metadata.get(
        "page_number"
    )

    if page_number is not None:
        return page_number

    original_page = document.metadata.get(
        "page"
    )

    if original_page is not None:
        return int(original_page) + 1

    return "Unknown"


# الفانكشن دي بتجمع الأجزاء المسترجعة في نص منظم يتبعت للنموذج.
def build_context(
    documents: List[Document]
) -> str:
    """بجمع الأجزاء المسترجعة في نص واحد منظم."""

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):
        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page_number = get_page_number(
            document
        )

        context_part = (
            f"[Context {index}]\n"
            f"Source: {source}\n"
            f"Page: {page_number}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

        context_parts.append(
            context_part
        )

    return "\n\n".join(
        context_parts
    )


# الفانكشن دي بتطلع المصادر وأرقام الصفحات من النتائج من غير تكرار.
def get_sources(
    documents: List[Document]
) -> List[str]:
    """بطلع أسماء المصادر وأرقام الصفحات من غير تكرار."""

    sources = []

    # هنا بلف على النتائج واحدة واحدة.
    for document in documents:
        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page_number = get_page_number(
            document
        )

        source_reference = (
            f"{source}, page {page_number}"
        )

        if source_reference not in sources:
            sources.append(
                source_reference
            )

    return sources


# الفانكشن دي بتشغل المشروع كامل: استرجاع، تجهيز سياق، إرسال للنموذج، ثم إرجاع الإجابة والمصادر.
def generate_answer(
    question: str,
    k: int = 4
) -> Tuple[str, List[str]]:
    """بشغل الدورة كاملة من السؤال لحد الإجابة والمصادر."""

    if not question or not question.strip():
        raise ValueError(
            "The question cannot be empty."
        )

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY was not found. "
            "Check your .env file."
        )

    # هنا بجيب أنسب أجزاء للسؤال الأول.
    documents = retrieve_context(
        query=question.strip(),
        k=k
    )

    if not documents:
        return (
            "I could not find this information "
            "in the Digital Notes.",
            []
        )

    # هنا بجمع النتائج في شكل مرتب.
    context = build_context(
        documents
    )

    # هنا بحط السؤال والكلام المسترجع في الرسالة النهائية.
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question.strip()
    )

    # هنا بعمل الاتصال بالخدمة باستخدام المفتاح السري.
    client = Groq(
        api_key=GROQ_API_KEY
    )

    # هنا ببعت الطلب للنموذج وأستقبل الرد.
    response = (
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )
    )

    # هنا بستخرج نص الإجابة من رد الخدمة.
    answer = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    # هنا بجمع المصادر الفعلية من بيانات الأجزاء المسترجعة.
    sources = get_sources(
        documents
    )

    return answer, sources


# الفانكشن دي بتشغل اختبار الملف لما أشغله مباشرة من سطر الأوامر.
def main():
    """دي تجربة سريعة للملف لما أشغله لوحده."""

    try:
        question = input(
            "Ask True Guide: "
        ).strip()

        # هنا بشغل خط الاسترجاع والتوليد وأستقبل الإجابة والمصادر.
        answer, sources = generate_answer(
            question=question,
            k=4
        )

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        if sources:
            for source in sources:
                print(f"- {source}")
        else:
            print("No sources were retrieved.")

    except Exception as error:
        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)
        print(type(error).__name__)
        print(error)


if __name__ == "__main__":
    main()
