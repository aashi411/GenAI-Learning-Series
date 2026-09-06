
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter 
# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate

splitter = CharacterTextSplitter(separator= "", chunk_size=10, chunk_overlap=1)

data = TextLoader(r"Lesson2_RAG\document loaders\notes2.txt")
docs = data.load()
chunks= splitter.split_documents(docs)

# template= ChatPromptTemplate.from_messages([
#     ("system",  """
#  You are an expert text summarization assistant.

#  Read the given document carefully and generate a concise summary.

#  The summary should:
#  - Cover the main ideas.
#  - Preserve important facts.
#  - Be clear and easy to understand.
#  - Be around 150 words unless instructed otherwise.
#  """),
#     ("human", "{data}")
# ])

# model = ChatMistralAI(model = "mistral-small-2506")

# prompt = template.format_messages(data = docs[0].page_content)

# result= model.invoke(prompt)

# print(result.content)

print(len(chunks))
print(chunks)
for i in chunks:
    print(i.page_content)
    print()
    print()