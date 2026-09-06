
from langchain_huggingface import HuggingFaceEmbeddings
 
embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
)
texts = [
    "Hello this is aashi",
    "It is my first time generating embeddins",
    "You must be a free model"
]
vector_doc = embedding.embed_documents(texts)
print(vector_doc)