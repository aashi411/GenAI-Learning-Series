from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

model= ChatMistralAI(model= "mistral-small-2506")
parser= StrOutputParser()

short_prompt= ChatPromptTemplate.from_template(
    "Explain {topic} in short"
)
long_prompt= ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)
topic = "Machine Learning"

#define multiple purposes
chain=RunnableParallel({
    "short" : RunnableLambda(lambda x: x ['short']) | short_prompt | model | parser ,
    "detailed" : RunnableLambda(lambda x: x ['detailed']) | long_prompt | model | parser
})
#res= chain.invoke({"topic" : "Machine Learning"})
#to ip diff topic in the prompts
res = chain.invoke({
    "short": {"topic": "Machine Learning"},
    "detailed": {"topic": "Deep Learning"}
})
print(res['short'])
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(res['detailed'])