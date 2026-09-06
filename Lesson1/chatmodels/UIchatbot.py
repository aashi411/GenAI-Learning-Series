from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

st.set_page_config(page_title="Mood AI", page_icon="🎭", layout="centered")

# ---------------- Classy styling ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f0f14 0%, #1a1a24 100%);
        color: #e8e6e3;
    }
    .main-title {
        font-family: 'Georgia', serif;
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 1px;
        background: linear-gradient(90deg, #d4af37, #f2e0a1, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #9a9a9a;
        font-family: 'Georgia', serif;
        font-style: italic;
        margin-top: -8px;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 6px 4px;
    }
    section[data-testid="stSidebar"] {
        background: #14141c;
        border-right: 1px solid #2a2a35;
    }
    .stButton>button {
        background: linear-gradient(90deg, #d4af37, #b8860b);
        color: #14141c;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #f2e0a1, #d4af37);
        color: #14141c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODES = {
    "😠 Angry": "You are an angry Ai agent, response must be aggressive and impatient.",
    "😂 Funny": "You are an funny Ai agent, response must be unserious and comical.",
    "🌧️ Gloomy": "You are an gloomy Ai agent, response must be sad and depressing.",
}

# ---------------- Session state ----------------
if "mode_label" not in st.session_state:
    st.session_state.mode_label = None
if "messages" not in st.session_state:
    st.session_state.messages = []

model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

st.markdown('<div class="main-title">Mood AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Choose a personality, then start the conversation</div>', unsafe_allow_html=True)

# ---------------- Sidebar: mode selection ----------------
with st.sidebar:
    st.markdown("### Choose your AI mode")
    for label in MODES:
        if st.button(label, use_container_width=True):
            if st.session_state.mode_label != label:
                st.session_state.mode_label = label
                st.session_state.messages = [SystemMessage(content=MODES[label])]

    st.markdown("---")
    if st.session_state.mode_label:
        st.markdown(f"**Current mode:** {st.session_state.mode_label}")
    if st.button("Reset chat", use_container_width=True):
        st.session_state.mode_label = None
        st.session_state.messages = []
        st.rerun()

# ---------------- Main chat area ----------------
if not st.session_state.mode_label:
    st.info("👈 Pick a mode from the sidebar to begin.")
else:
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    prompt = st.chat_input("You: ")

    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)

        response = model.invoke(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

        with st.chat_message("assistant"):
            st.write(response.content)