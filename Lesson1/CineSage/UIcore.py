from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

st.set_page_config(page_title="CineSage", page_icon="🎬", layout="centered")

# ---------------- Galaxy theme styling ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 20%, #2d0b4e 0%, #1a0533 35%, #0d0221 100%);
        color: #f5e9ff;
    }
    .main-title {
        font-family: 'Georgia', serif;
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6ec7, #b967ff, #ff4d6d, #ffd166);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #d6b8ff;
        font-style: italic;
        margin-top: -6px;
        margin-bottom: 1.8rem;
    }
    .stTextArea textarea {
        background-color: rgba(45, 11, 78, 0.55);
        color: #f5e9ff;
        border: 1px solid #b967ff;
        border-radius: 12px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4d6d, #b967ff, #ffd166);
        color: #0d0221;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ffd166, #ff6ec7, #b967ff);
        color: #0d0221;
    }
    .result-box {
        background: rgba(45, 11, 78, 0.45);
        border: 1px solid #ff6ec7;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert movie analyst and information extraction assistant.

            Your task is to analyze the given movie description carefully and extract all meaningful information.

            For every movie mentioned, provide the following:

            - Movie Name
            - Genre
            - Main Cast (if mentioned, otherwise say "Not Mentioned")
            - Director (if mentioned)
            - Release Year (if mentioned)
            - Main Characters
            - Setting
            - Main Themes
            - Central Conflict
            - Key Plot Points
            - Important Keywords
            - A short summary (2-3 sentences)

            Guidelines:
            - If multiple movies are present, analyze each one separately.
            - Do not invent facts that are not present in the paragraph.
            - If any information is unavailable, clearly state "Not Mentioned".
            - Present the output in a neat, readable format using headings and bullet points.
            """,
        ),
        (
            "human",
            """
                Analyze the following paragraph:

                {paragraph}
            """,
        ),
    ]
)

st.markdown('<div class="main-title">🎬 CineSage</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Extract structured insights from any movie description</div>', unsafe_allow_html=True)

para = st.text_area("Give your paragraph:", height=200)

if st.button("Analyze"):
    if para.strip():
        final_prompt = prompt.invoke({"paragraph": para})
        response = model.invoke(final_prompt)
        st.markdown(f'<div class="result-box">{response.content}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a paragraph to analyze.")