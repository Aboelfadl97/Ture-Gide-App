"""
streamlit_app.py

Streamlit interface for the True Guide RAG project.
The application uses Groq to generate answers from
the retrieved Digital Notes context.
"""

from importlib import import_module

import streamlit as st


# Import 07_prompting.py dynamically because
# the filename starts with a number.
prompting_module = import_module(
    "07_prompting"
)

generate_answer = (
    prompting_module.generate_answer
)

GROQ_MODEL = (
    prompting_module.GROQ_MODEL
)

GROQ_API_KEY = (
    prompting_module.GROQ_API_KEY
)


st.set_page_config(
    page_title="True Guide RAG",
    page_icon="📘",
    layout="centered",
)


st.title("📘 True Guide")

st.caption(
    "Ask questions from the "
    "Data Analytics Digital Notes."
)


# Sidebar
with st.sidebar:
    st.header("Project Information")

    st.markdown(
        """
        **True Guide** uses:

        - Digital Notes PDF
        - LangChain
        - Hugging Face Embeddings
        - ChromaDB
        - Groq
        - Streamlit
        """
    )

    st.divider()

    st.subheader("Language Model")

    st.code(
        GROQ_MODEL
    )

    st.divider()

    top_k = st.slider(
        label="Number of retrieved chunks",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
    )

    st.caption(
        "This controls how many relevant "
        "text chunks are retrieved from ChromaDB."
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


# Initialize chat history
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


# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


# User question
question = st.chat_input(
    "Example: What is ethics and privacy?"
)


if question:
    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner(
            "Searching the Digital Notes..."
        ):
            try:
                answer, sources = generate_answer(
                    question=question,
                    k=top_k,
                )

                st.markdown(answer)

                if sources:
                    with st.expander(
                        "Retrieved Sources"
                    ):
                        for source in sources:
                            st.markdown(
                                f"- {source}"
                            )

                assistant_message = answer

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

    # Save assistant response
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