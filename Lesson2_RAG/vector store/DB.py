# not gonna run as it will use open ai which is paid

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document 

docs =[
    Document(page_content= "Python is widely used in Artificial Intelligence.", metadata= {"source": "AI_book"}),
    Document(page_content= "Pandas is used for data analysis in python.", metadata= {"source": "DataScience_book"}),
    Document(page_content= "Neural networks are used in deep Learning.", metadata= {"Source": "DL_book"}),
]

embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model, 
    persist_directory= 'chroma-db'#will make a folder named chroma-db
)

result = vectorstore.similarity_search("what is used for data analysis?", k=2)# k= no. of doc/embeddings/chumks to be returned from the search

for r in result:
    print(r.page_content)
    print(r.metadata)
    # will return same doc 2 times as 2 time they are searching the same doc twice
    #to avoid this delete the chroma-db file once 

retriever = vectorstore.as_retriever()
docs = retriever.invoke("Explain deep learning")
for d in docs:
    print(d.page_content)
