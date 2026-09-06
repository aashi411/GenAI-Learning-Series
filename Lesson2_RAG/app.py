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

.stApp{
    background-color:#2B0B1B;
}

h1,h2,h3,p,label{
    color:white;
}

div[data-testid="stChatMessage"]{
    background-color:#4A162E;
    border-radius:15px;
    padding:10px;
}

.stButton>button{
    background:#7A2948;
    color:white;
    border-radius:10px;
    width:100%;
}

.stButton>button:hover{
    background:#9B3D63;
}

section[data-testid="stSidebar"]{
    background:#3A1025;
}

</style>
""",unsafe_allow_html=True)

# ----------------------------------------------------
# Session State
# ----------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages=[]

if "db_created" not in st.session_state:
    st.session_state.db_created=False

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

with st.sidebar:

    st.title("📚 StudyCat")

    st.write("Upload your study material.")

    uploaded_pdf=st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    if st.button("📖 Build Knowledge Base"):

        if uploaded_pdf is None:
            st.warning("Please upload a PDF.")
        else:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(uploaded_pdf.read())
                temp_path=tmp.name

            with st.spinner("Reading your book..."):

                pages,chunks=create_vector_db(temp_path)

            os.remove(temp_path)

            st.success("Knowledge Base Created!")

            st.write(f"📄 Pages : {pages}")
            st.write(f"🧩 Chunks : {chunks}")

            st.session_state.db_created=True

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages=[]

# ----------------------------------------------------
# Main Page
# ----------------------------------------------------

st.title("📚 StudyCat RAG Assistant")

st.caption("Upload a book and ask questions from it.")

# ----------------------------------------------------
# Chat History
# ----------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ----------------------------------------------------
# Chat Input
# ----------------------------------------------------

question=st.chat_input("Ask something about your book...")

if question:

    if not st.session_state.db_created:

        st.warning("Please upload a PDF and build the knowledge base first.")

    else:

        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Searching your book..."):

            answer,docs=ask_question(question)

        with st.chat_message("assistant"):

            st.markdown(answer)

            with st.expander("📑 Retrieved Context"):

                for i,doc in enumerate(docs):

                    st.markdown(f"### Chunk {i+1}")

                    st.write(doc.page_content)

                    st.divider()

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )