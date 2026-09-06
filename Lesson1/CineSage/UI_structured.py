from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from typing import List, Optional
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🎬 Movie Information Extractor",
    page_icon="🎥",
    layout="wide"
)

# -----------------------------
# LLM
# -----------------------------
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0
)

# -----------------------------
# Pydantic Schema
# -----------------------------
class Movie(BaseModel):
    title: str
    release_yr: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert movie information extraction assistant.

Extract the movie information from the paragraph.

{format_instructions}
            """,
        ),
        ("human", "{paragraph}"),
    ]
)

# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Information Extractor")

st.write(
    "Paste a movie description below and extract useful information using **LangChain + Mistral AI**."
)

paragraph = st.text_area(
    "Movie Description",
    height=250,
    placeholder="Paste your paragraph here..."
)

if st.button("Extract Information", use_container_width=True):

    if paragraph.strip() == "":
        st.warning("Please enter a paragraph.")
        st.stop()

    with st.spinner("Extracting information..."):

        final_prompt = prompt.invoke(
            {
                "paragraph": paragraph,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        response = model.invoke(final_prompt)

        movie = parser.parse(response.content)

    st.success("Extraction Complete!")

    st.subheader("Movie Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🎬 Title", movie.title)

        st.write("**Release Year**")
        st.write(movie.release_yr)

        st.write("**Director**")
        st.write(movie.director)

        st.write("**IMDb Rating**")
        st.write(movie.rating)

    with col2:
        st.write("**Genres**")
        st.write(", ".join(movie.genre))

        st.write("**Cast**")

        if movie.cast:
            for actor in movie.cast:
                st.write(f"• {actor}")
        else:
            st.write("Not Mentioned")

    st.divider()

    st.subheader("Summary")

    st.info(movie.summary)

    st.divider()

    with st.expander("View JSON Output"):

        st.json(movie.model_dump())