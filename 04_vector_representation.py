"""الملف ده بحمل فيه الموديل اللي بيحول الكلام لأرقام."""

# الموديل بيحول الكلام لأرقام علشان نقارن المعنى.

# هنا بجيب الأداة اللي هتحمل الموديل.
from langchain_huggingface import HuggingFaceEmbeddings


# هنا بكتب اسم الموديل اللي هنستخدمه.
DEFAULT_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# الفانكشن دي بتحمل نموذج التمثيل العددي وبترجعه جاهز للاستخدام.
def get_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME
):
    """بحمل الموديل اللي بيحول النص لأرقام."""

    # هنا بحمل الموديل على المعالج وبخلي الأرقام في نفس المقياس.
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embedding_model


# الفانكشن دي بستخدمها علشان أجرب الملف لوحده
# وأتأكد إن الـ Embedding شغال قبل ما أوصله بباقي المشروع
def main():

    try:

        # بطبع عنوان علشان الأوتبوت يبقى واضح
        print("=" * 60)
        print("Loading Embedding Model...")
        print("=" * 60)

        # هنا بحمل الـ Embedding Model
        embedding_model = get_embedding_model()

        # جملة بسيطة أجرب عليها
        test_text = (
            "Data analytics helps organizations "
            "make better decisions."
        )

        # هنا بحول الجملة لـ Vector
        vector = embedding_model.embed_query(
            test_text
        )

        # لو وصلت هنا يبقى الموديل اشتغل تمام
        print(
            "Embedding Model Loaded Successfully"
        )

        print("-" * 60)

        # بطبع اسم الموديل اللي شغال بيه
        print("Model Name:")
        print(DEFAULT_MODEL_NAME)

        print("-" * 60)

        # بطبع عدد أبعاد الـ Vector
        print(
            f"Vector Dimensions: "
            f"{len(vector)}"
        )

        print("-" * 60)

        # بطبع أول 10 قيم بس
        # علشان أتأكد إن الـ Vector اتعمل
        print(
            "First 10 vector values:"
        )

        print(
            vector[:10]
        )

        print("=" * 60)

    except Exception as error:

        # لو حصل أي Error بطبعه
        # علشان أعرف المشكلة فين
        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(
            type(error).__name__
        )

        print(error)


# السطر ده معناه:
# لو شغلت الملف ده لوحده هينفذ main()
# لكن لو الملف اتعمله import في ملف تاني
# مش هينفذها
if __name__ == "__main__":
    main()