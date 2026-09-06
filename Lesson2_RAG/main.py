from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_openai import OpenAIEmbeddings #we'll use hugging face
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatMistralAI(model = "mistral-small-2506")

#prompt template
prompt= ChatPromptTemplate.from_messages(# from messages bc its easier to define roles there.
    [
        ("system",
            """
            You are a helpful AI assistant. 
            use ONLY the provided context to answer the question.
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
print("press 0 to exit ")

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

def ask_question(query):

    #embedding_model = OpenAIEmbeddings()
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorestore = Chroma(
        persist_directory= "chroma_db",
        embedding_function= embedding_model
    )
    retriver = vectorestore.as_retriever(
        search_type = "mmr",
        search_kwargs ={ #kwargs- keywords arguments
            "k" : 4, # retrive 4 results
            "fetch_k" : 10, #first find 10 by similarity search and then from them apply mmr and find best 4
            "lambda_mult" : 0.5 # 0- bahut zada diverse results, 1- bahut kam diverse results 
        } 
    )

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
    response = llm.invoke(final_prompt)

    return response.content, docs