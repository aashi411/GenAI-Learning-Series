import streamlit as st
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

# ---------------------------
# LangChain Components
# ---------------------------
search_tool = TavilySearchResults(max_results=5)

llm = ChatMistralAI(
    model="mistral-small-2506"
)

prompt = ChatPromptTemplate.from_template(
    """
You are an AI News Curator.

Summarize the following news into:
- Short bullet points
- Mention important companies
- Mention major announcements
- Keep each point concise

News:
{news}
"""
)

chain = prompt | llm | StrOutputParser()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="AI News Curator",
    page_icon="📰",
    layout="wide"
)

st.title("📰 AI News Curator")
st.write("Get summarized AI news powered by **Tavily + Mistral AI + LangChain**")

query = st.text_input(
    "Search Topic",
    value="Latest AI news of 2026"
)

col1, col2 = st.columns([1, 4])

with col1:
    fetch = st.button("Fetch News")

if fetch:

    with st.spinner("Searching latest news..."):

        try:
            news = search_tool.run(query)

            summary = chain.invoke(
                {
                    "news": news
                }
            )

            st.success("News Retrieved!")

            tab1, tab2 = st.tabs(["Summary", "Raw Search Results"])

            with tab1:
                st.markdown(summary)

            with tab2:
                st.write(news)

        except Exception as e:
            st.error(str(e))