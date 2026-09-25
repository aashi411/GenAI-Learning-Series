from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_openai import OpenAIEmbeddings #we'll use hugging face
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
#from langchain_mistralai import ChatMistralAI
#from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import time

start_time = time.time()

llm = ChatGroq(model = "openai/gpt-oss-120b", temperature=0.7)
#llm = ChatOllama(model = "gemma3:1b", temperature=0.2)
#prompt template
prompt= ChatPromptTemplate.from_messages(# from messages bc its easier to define roles there.
    [
        ("system",
            """
            You are a helpful AI assistant. 
            Simplify the language in answer also provide some basic exampe from real world to explain.
            If the answer is not present in the context,
            say: "I could not find the answer in the document."
            """
         ), ("human", """
                Context: {context}
                Question: {question}    
            """

             )
    ]
)

print("Rag ystem Created")
#print("press 0 to exit ")

# while True:
#     query=input("You: ")
#     if query == "0":
#         break 
#     docs = retriver.invoke(query)

#     context= "\n\n".join( # \n\n is for giving some space on top
#         [doc.page_content for doc in docs ]
#         # loop for taking the page content of every document and then saving in the list and then joining then
#         # so it becomes a single string i.e. context 
#     )

#     final_prompt = prompt.invoke({
#         "context" : context,
#         "question" : query
#     })

#     response= llm.invoke(final_prompt)

#     print(f"\n AI : {response.content}")

#for streamlit application

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()
print(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")

@st.cache_resource
def load_vectorstore(_embedding_model):
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=_embedding_model
    )
#The _ before embedding_model tells Streamlit not to try to hash that object.
vectorestore = load_vectorstore(embedding_model)
print(f"Vector store loaded in {time.time() - start_time:.2f} seconds")


#Maximum Marginal Relevance- mmr
@st.cache_resource
def load_retriever(_vectorstore):
    return _vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs ={ #kwargs- keywords arguments
            "k" : 3, # retrive 3 results
            "fetch_k" : 7, #first find 10 by similarity search and then from them apply mmr and find best 4
            "lambda_mult" : 0.5 # 0- bahut zada diverse results, 1- bahut kam diverse results 
        } 
    )
retriver = load_retriever(vectorestore)
print(f"Retriever loaded in {time.time() - start_time:.2f} seconds")

def ask_question(query):

    #embedding_model = OpenAIEmbeddings()
    
    # Retrieve relevant documents
    docs = retriver.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })


    #temp 

    print("=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}")
        print(doc.metadata)
        print(doc.page_content[:500])

    # Get answer from LLM
    response = llm.invoke(final_prompt)

    return response.content, docs
