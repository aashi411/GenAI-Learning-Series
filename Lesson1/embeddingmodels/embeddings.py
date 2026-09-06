from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings

embeddings= OpenAIEmbeddings(
    model = 'text-embedding-3-large',
    dimensions = 64
)
texts = [
    "Hello hihuh",
    "yoi sgfsshvj",
    "Igjshwbjguwg jbshvh"
]
vector_doc = embeddings.embed_documents(texts)
vector =embeddings.embed_query("You are going to laern GEN AI")
print(vector_doc)
print(vector)
#it is also paid so vont run this 