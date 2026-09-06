from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_result= 5) #max 5 results for less tokens to  burn
# llm = ChatMistralAI(model= "mistral-small-2506")
# prompt = ChatPromptTemplate.from_template(
#     """"
#         You are a helpful assistents.
#         Summarize the following news into clear bullet points
#         {news}
#     """
# )
# chain = prompt | llm | StrOutputParser()
# # chain = news_res | prompt | llm | StrOutputParser() will also work
# news_res= search_tool.run("Latest AI news of 2026")#query

# res = chain.invoke({"news": news_res})

#print(res)

print(search_tool.description)
print(search_tool.name)
print(search_tool.args)