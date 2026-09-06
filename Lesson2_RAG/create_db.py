# #load pdf
# #split into chunks
# # create the embeddings
# #store into chroma 



# Load PDF
# Split into chunks
# Create embeddings (Local - FREE)
# Store embeddings in ChromaDB

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_db(pdf_path):
    # Load PDF
    #loader = PyPDFLoader(r"Lesson2_RAG\document loaders\deep-learning.pdf")
    loader = PyPDFLoader(pdf_path)#for streamlit app
    docs = loader.load()

    print(f"Loaded {len(docs)} pages.")

    # Split into Chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks.")

    # Local Embedding Model
    # (Uses cached model if already downloaded)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Chroma Vector Store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    print(" Chroma Vector Database created successfully!")

    return len(docs), len(chunks)



# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from dotenv import load_dotenv
# load_dotenv()

# data = PyPDFLoader(r"Lesson2_RAG\document loaders\deep-learning.pdf")
# docs = data.load()
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 200
# )

# chunks = splitter.split_documents(docs)

# embedding_model = OpenAIEmbeddings()

# vectorestore = Chroma.from_documents(
#     documents= chunks,
#     embedding= embedding_model,
#     persist_directory= "chroma_db"
# )

