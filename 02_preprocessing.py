"""الملف ده بنضف فيه النص وبجهزه للبحث."""

# الملف ده فيه تنظيف النص قبل التقسيم والبحث.
# فيه تنظيف خفيف للنص، وتجهيز منفصل للكلمات المستخدمة مع البحث بالكلمات.

# هنا بجيب أدوات التعبيرات المنتظمة المستخدمة في تنظيف النص.
import re

# هنا بجيب طريقة تسمحلي أحمّل ملفات بايثون أسماؤها بتبدأ بأرقام.
from importlib import import_module

# هنا بحدد شكل القوائم علشان بايثون يبقى فاهم نوع البيانات.
from typing import List

# هنا بجيب شكل المستند اللي بيحفظ النص ومعاه بيانات الصفحة.
from langchain_core.documents import Document


# الفانكشن دي بتنضف النص من مشاكل القراءة من غير ما تغير المعنى.
def clean_text(text: str) -> str:
    """بنضف النص من مشاكل الاستخراج من غير ما نغير معناه."""

    # لو النص فاضي برجع نتيجة فاضية وخلاص.
    if not text:
        return ""


    # هنا بشيل الحروف الخفية اللي ممكن تظهر بسبب استخراج البي دي إف.
    text = text.replace("\x00", " ")


    # هنا بشيل علامة الشرط الناعم اللي ساعات تقسم الكلمة من غير ما تظهر.
    text = text.replace("\u00ad", "")


    # هنا بنضف الجزء ده بقاعدة منتظمة.
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )


    # هنا بنضف الجزء ده بقاعدة منتظمة.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )


    # هنا بنضف الجزء ده بقاعدة منتظمة.
    text = re.sub(
        r" *\n *",
        "\n",
        text
    )


    # هنا بنضف الجزء ده بقاعدة منتظمة.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# الفانكشن دي بتفصل النص لكلمات علشان البحث يشتغل عليها.
def tokenize_for_bm25(text: str) -> List[str]:
    """بحول النص لقائمة كلمات علشان البحث بالكلمات."""

    # لو النص فاضي برجع نتيجة فاضية وخلاص.
    if not text:
        return []

    # هنا بخلي الحروف كلها صغيرة علشان البحث مايفرقش بينهم.
    text = text.lower()

    # هنا باخد الكلمات والأرقام بس، وبسيب علامات الترقيم.
    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text
    )

    return tokens


# الفانكشن دي بتنضف كل الصفحات وبتسيب بياناتها زي ما هي.
def preprocess_documents(
    documents: List[Document]
) -> List[Document]:
    """بطبق التنظيف على كل الصفحات وبحافظ على بياناتها."""

    # هنا بعمل قائمة الصفحات بعد التنظيف.
    cleaned_documents = []

    # هنا بلف على النتائج واحدة واحدة.
    for document in documents:

        # هنا بنضف نص الصفحة الحالية.
        cleaned_text = clean_text(
            document.page_content
        )


        # لو الصفحة طلعت فاضية بعد التنظيف بتخطاها.
        if not cleaned_text:
            continue

        # هنا بعمل نسخة بالنص النضيف وبنفس بيانات الصفحة.
        cleaned_document = Document(
            page_content=cleaned_text,
            metadata=document.metadata.copy()
        )

        cleaned_documents.append(
            cleaned_document
        )

    return cleaned_documents


# الفانكشن دي بتش# هنا بجرب مرحلة الـ Preprocessing لوحدها
# علشان أتأكد إن التنضيف شغال
# وإن الـ Metadata متأثرتشغل اختبار الملف لما أشغله مباشرة من سطر الأوامر.

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

        print("=" * 60)

        print(
            f"Original PDF Pages: "
            f"{len(documents)}"
        )

        print(
            f"Cleaned PDF Pages: "
            f"{len(cleaned_documents)}"
        )

        print("=" * 60)

        if cleaned_documents:

            first_document = (
                cleaned_documents[0]
            )

            print("\nLIGHT PREPROCESSING")
            print("-" * 60)

            print(
                first_document.page_content[:1000]
            )

            print("\n" + "=" * 60)

            print("BM25 TOKENS")
            print("-" * 60)

            tokens = tokenize_for_bm25(
                first_document.page_content
            )

            print(tokens[:50])

            print("\n" + "=" * 60)

            print(
                "Source:",
                first_document.metadata.get(
                    "source",
                    "Unknown"
                )
            )

            print(
                "Page:",
                first_document.metadata.get(
                    "page",
                    "Unknown"
                )
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
