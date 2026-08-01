"""الملف ده فيه واجهة التطبيق اللي المستخدم بيتعامل معاها."""

# الملف ده فيه شكل التطبيق اللي المستخدم بيفتحه.
# المستخدم بيسأل، والنظام بيعرض الإجابة والمصادر والإعدادات النهائية.

# هنا بجيب أدوات التعامل مع متغيرات البيئة ومفتاح الخدمة.
import os

# هنا بجيب طريقة تسمحلي أحمّل ملفات بايثون أسماؤها بتبدأ بأرقام.
from importlib import import_module

# هنا بجيب مكتبة بناء واجهة الويب.
import streamlit as st


# هنا بظبط اسم الصفحة والأيقونة وشكلها.
st.set_page_config(
    page_title="True Guide RAG",
    page_icon="📘",
    layout="centered",
)


try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = (
            st.secrets["GROQ_API_KEY"]
        )

    if "GROQ_MODEL" in st.secrets:
        os.environ["GROQ_MODEL"] = (
            st.secrets["GROQ_MODEL"]
        )

except Exception:
    pass


prompting_module = import_module(
    "07_prompting"
)

generate_answer = (
    prompting_module.generate_answer
)

# هنا بكتب اسم الموديل اللي هيطلع الإجابة.
GROQ_MODEL = (
    prompting_module.GROQ_MODEL
)

# هنا باخد المفتاح من بره الكود علشان مايبقاش مكشوف.
GROQ_API_KEY = (
    prompting_module.GROQ_API_KEY
)


# هنا بثبت حجم الجزء اللي كسب في التقييم.
FINAL_CHUNK_SIZE = 300

# هنا بثبت التداخل اللي كسب في التقييم.
FINAL_CHUNK_OVERLAP = 50

# هنا بثبت طريقة البحث اللي طلعت أحسن.
FINAL_RETRIEVER = "Semantic"

# هنا بثبت عدد النتائج اللي هنرجعها.
FINAL_TOP_K = 2


st.title("📘 True Guide")

st.caption(
    "Ask questions from the "
    "Data Analytics Digital Notes."
)


# هنا بعمل الجزء الجانبي اللي فيه معلومات المشروع.
with st.sidebar:

    st.header(
        "Project Information"
    )

    st.markdown(
        """
        **True Guide uses:**

        - Digital Notes PDF
        - LangChain
        - Hugging Face Embeddings
        - ChromaDB
        - Groq
        - Streamlit
        """
    )

    st.divider()

    st.subheader(
        "Final Evaluated Configuration"
    )

    st.write(
        f"**Chunk Size:** "
        f"{FINAL_CHUNK_SIZE}"
    )

    st.write(
        f"**Chunk Overlap:** "
        f"{FINAL_CHUNK_OVERLAP}"
    )

    st.write(
        f"**Retriever:** "
        f"{FINAL_RETRIEVER}"
    )

    st.write(
        f"**Top-K:** "
        f"{FINAL_TOP_K}"
    )

    st.caption(
        "These settings were selected "
        "after experimental evaluation."
    )

    st.divider()

    st.subheader(
        "Language Model"
    )

    st.code(
        GROQ_MODEL
    )

    st.divider()

    if GROQ_API_KEY:

        st.success(
            "Groq API key loaded successfully."
        )

    else:

        st.error(
            "GROQ_API_KEY was not found."
        )


# هنا بعمل ذاكرة المحادثة أول مرة المستخدم يفتح التطبيق.
if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! Ask me a question "
                "from the Data Analytics "
                "Digital Notes."
            ),
        }
    ]


# هنا بعرض الرسائل اللي اتقالت في نفس الجلسة.
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# هنا بعمل مربع كتابة يستقبل سؤال المستخدم.
question = st.chat_input(
    "Example: What is data analytics?"
)


# هنا ببدأ التنفيذ لما المستخدم يكتب سؤال فعلي.
if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the Digital Notes..."
        ):

            try:

                # هنا بشغل خط الاسترجاع والتوليد وأستقبل الإجابة والمصادر.
                answer, sources = (
                    generate_answer(
                        question=question,
                        k=FINAL_TOP_K,
                    )
                )

                st.markdown(
                    answer
                )

                if sources:

                    with st.expander(
                        "Retrieved Sources"
                    ):

                        for source in sources:

                            st.markdown(
                                f"- {source}"
                            )

                assistant_message = (
                    answer
                )

                if sources:

                    assistant_message += (
                        "\n\n**Sources:**\n"
                    )

                    for source in sources:

                        assistant_message += (
                            f"- {source}\n"
                        )

            except Exception as error:

                assistant_message = (
                    "An error occurred while "
                    "generating the answer:\n\n"
                    f"`{type(error).__name__}: "
                    f"{error}`"
                )

                st.error(
                    assistant_message
                )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_message,
        }
    )


st.divider()

st.caption(
    "True Guide answers using retrieved "
    "context from the Digital Notes PDF."
)
