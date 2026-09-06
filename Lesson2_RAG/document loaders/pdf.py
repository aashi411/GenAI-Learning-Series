from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader(r"Lesson2_RAG\document loaders\QT-SAT.pdf")


docs= data.load()
print(docs)
print(len(docs))
print(docs[6]) #will give the data of the 6th pages