#Chat-prompt templates
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI (model= 'mistral-small-2506')


# Instantiation using from_template (recommended)
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

para = input("Give your paragraph: ")

final_prompt= prompt.invoke(
    {"paragraph": para}
)

response = model.invoke(final_prompt)

print(response.content)