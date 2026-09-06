from dotenv import load_dotenv
load_dotenv()

import os
import requests #used to hit an url online

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent 
from langchain.agents.middleware  import wrap_tool_call

#Weather tool
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    api_key= os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response= requests.get(url)
    data= response.json()
    print("Debug: ", data)
    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not featch weather')} "
    temp= data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}*C"

print(get_weather.invoke("Bhopal"))


#tavily news tool
tavily_client= TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str)-> str:
    """Get latest news  about the city"""
    response=tavily_client.search(
        query= f"latest news in {city}",
        search_depth= "basic",
        max_results=3
    )
    results= response.get("results", [])
    if not results:
        return f"NO news found for {city}"


    news_list= []
    for r in results:
        title= r.get("title", "No title")
        url = r.get("url", "")
        snippet= r.get("content", "")

        news_list.append(
            f"- {title}\n {snippet[:100]}..."
        )

    return f"Latest news in {city}:\n\n "+ "\n\n".join(news_list)



llm = ChatMistralAI(model= "mistral-small-2506")

#middle-ware
@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call"""
    tool_name= request.tool_call["name"]
    confirm= input(f"Agent wants to call `{tool_name}. Approve? (yes/No)`")

    if confirm.lower()== "yes":
        return ToolMessage(
            content= "Tool call denied by user.",
            tool_call_id= request.tool_call["id"]
        )
    return handler(request)

#agent creation
agent = create_agent(
    llm,
    tools= [get_weather, get_weather],
    system_prompt="you are a helpful city assistent.",
    middleware= [human_approval]
)
print("CITY AGENT")
print("Type EXIT to quit")
while True:
    user_input= input("You: ")
    if user_input.lower() == "exit":
        break

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })
    print(result['messages'][-1].content)