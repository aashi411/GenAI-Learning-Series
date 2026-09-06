from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


model= ChatMistralAI(model= "mistral-small-2506")
parser= StrOutputParser()
code_prompt= ChatPromptTemplate.from_messages([
    ("system", "You are code generator"),
    ("human", "{topic}")
])

explain_prompt= ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistemt who explains code in simple"),
    ("human", "explain the following code in simle words: \n{code}")
])
# seq= code_prompt | model | parser | explain_prompt | model | parser
# #here i wont get the code directly to the explainiation 
# res =seq.invoke({"topic": "write a code of palindrome in python"})
# print(res)

seq= code_prompt | model | parser
seq2 = RunnableParallel(
    {
        "code": RunnablePassthrough(), #will return what ever comes to it as it was
        "explaination" : explain_prompt | model | parser
    }
)

chain= seq | seq2
res= chain.invoke({"topic" : "write a code of palindrome in python"})
print(res['code'])
print(res['explaination'])