import streamlit as st
import tempfile
import os

from create_db import create_vector_db
from main import ask_question


# ----------------------------------------------------
# Page Config
# ----------------------------------------------------

st.set_page_config(
    page_title="📚 StudyCat RAG",
    page_icon="📖",
    layout="wide"
)


# ----------------------------------------------------
# CSS
# ----------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #2B0B1B;
}

h1, h2, h3, p, label {
    color: white;
}

/* Chat messages */

div[data-testid="stChatMessage"] {
    background-color: #4A162E;
    border-radius: 15px;
    padding: 10px;
}

/* Buttons */

.stButton > button {
    background: #7A2948;
    color: white;
    border-radius: 10px;
    width: 100%;
    border: none;
}

.stButton > button:hover {
    background: #9B3D63;
    color: white;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #3A1025;
}

/* File uploader */

section[data-testid="stFileUploaderDropzone"] {
    background-color: #4A162E;
    border-radius: 10px;
}

/* Text input */

div[data-baseweb="input"] {
    background-color: #4A162E;
}

/* Chat input */

div[data-testid="stChatInput"] {
    background-color: #3A1025;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# Session State
# ----------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_created" not in st.session_state:
    st.session_state.db_created = os.path.exists("chroma_db")


# ----------------------------------------------------
# Sidebar - Knowledge Base
# ----------------------------------------------------

with st.sidebar:

    st.title("📚 StudyCat")

    st.write("Upload your study material.")

    # Knowledge Base Status
    if st.session_state.db_created:

        st.success("✅ Knowledge Base Ready")

        st.caption(
            "Your existing knowledge base is loaded."
        )

    else:

        st.info("📚 No knowledge base loaded yet.")


    # ------------------------------------------------
    # PDF FORM
    # ------------------------------------------------

    with st.form("knowledge_base_form"):

        uploaded_pdf = st.file_uploader(
            "Choose PDF",
            type=["pdf"]
        )

        build_database = st.form_submit_button(
            "📖 Build Knowledge Base",
            use_container_width=True
        )


    # ------------------------------------------------
    # Build Knowledge Base
    # ------------------------------------------------

    if build_database:

        if uploaded_pdf is None:

            if st.session_state.db_created:

                st.info(
                    "Using the existing knowledge base."
                )

            else:

                st.warning(
                    "Please upload a PDF first."
                )

        else:

            temp_path = None

            try:

                # ------------------------------------
                # Save uploaded PDF temporarily
                # ------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp.write(
                        uploaded_pdf.getvalue()
                    )

                    temp_path = tmp.name


                # ------------------------------------
                # Build Vector Database
                # ------------------------------------

                with st.spinner(
                    "📖 Reading and processing your book..."
                ):

                    pages, chunks = create_vector_db(
                        temp_path
                    )


                # ------------------------------------
                # Update state
                # ------------------------------------

                st.session_state.db_created = True

                st.success(
                    "✅ Knowledge Base Created!"
                )

                st.write(
                    f"📄 Pages: {pages}"
                )

                st.write(
                    f"🧩 Chunks: {chunks}"
                )


            finally:

                # ------------------------------------
                # Delete temporary PDF
                # ------------------------------------

                if temp_path and os.path.exists(temp_path):

                    os.remove(temp_path)


    st.divider()


    # ------------------------------------------------
    # Clear Chat
    # ------------------------------------------------

    if st.button(
        "🗑 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ----------------------------------------------------
# Main Page
# ----------------------------------------------------

st.title("📚 StudyCat RAG Assistant")

st.caption(
    "Upload a book and ask questions from it."
)


# ----------------------------------------------------
# Chat Section
# ----------------------------------------------------

@st.fragment
def chat_section():

    # -----------------------------------------------
    # Display Chat History
    # -----------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # -----------------------------------------------
    # Chat Input
    # -----------------------------------------------

    question = st.chat_input(
        "Ask something about your book..."
    )


    # -----------------------------------------------
    # Process Question
    # -----------------------------------------------

    if question:

        # -------------------------------------------
        # Check Knowledge Base
        # -------------------------------------------

        if not st.session_state.db_created:

            st.warning(
                "Please upload a PDF and build "
                "the knowledge base first."
            )

            return


        # -------------------------------------------
        # Add User Message
        # -------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message("user"):

            st.markdown(question)


        # -------------------------------------------
        # Ask RAG System
        # -------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching your book..."
            ):

                answer, docs = ask_question(
                    question
                )


            # ---------------------------------------
            # Display Answer
            # ---------------------------------------

            st.markdown(answer)


            # ---------------------------------------
            # Retrieved Context
            # ---------------------------------------

            with st.expander(
                "📑 Retrieved Context"
            ):

                if docs:

                    for i, doc in enumerate(docs):

                        st.markdown(
                            f"### Chunk {i + 1}"
                        )

                        st.write(
                            doc.page_content
                        )

                        st.divider()

                else:

                    st.write(
                        "No retrieved documents."
                    )


        # -------------------------------------------
        # Save Assistant Message
        # -------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ----------------------------------------------------
# Run Chat Fragment
# ----------------------------------------------------

chat_section()
