# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter

# data = PyPDFLoader(r"Lesson2_RAG\document loaders\QT-SAT.pdf")
# docs= data.load()
# splitter = TokenTextSplitter(
#     chunk_size=1000,
#     chunk_overlap= 10
# )
# chunks = splitter.split_documents(docs)
# print(len(chunks))

#now we'll use recursive text splitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader(r"Lesson2_RAG\document loaders\QT-SAT.pdf")
docs= data.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap= 10
)
chunks = splitter.split_documents(docs)
print(len(chunks))