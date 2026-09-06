from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
data = TextLoader(r"Lesson2_RAG\document loaders\notes.txt",
    encoding="utf-8")
docs = data.load()
template= ChatPromptTemplate.from_messages([
    ("system",  """
 You are an expert text summarization assistant.

 Read the given document carefully and generate a concise summary.

 The summary should:
 - Cover the main ideas.
 - Preserve important facts.
 - Be clear and easy to understand.
 - Be around 150 words unless instructed otherwise.
 """),
    ("human", "{data}")
])

model = ChatMistralAI(model = "mistral-small-2506")

prompt = template.format_messages(data = docs[0].page_content)

result= model.invoke(prompt)

print(result.content)