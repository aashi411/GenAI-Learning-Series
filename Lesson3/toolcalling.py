from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage #so that we can maintain a history of messages to give to llm for context
from rich import print

#1 creating a tool
@tool
def get_text_length(text: str)-> int:
    """
    Returns the number of character in a given text
    """
    return len(text)

#a binding dictonary for converting string to fn calls 
tools= {
    "get_text_length": get_text_length
}

llm = ChatMistralAI(model= "mistral-small-2506")

#2 tool binding
llm_with_tool= llm.bind_tools([get_text_length])
# res= llm.invoke("hello- 'why are you'")
# print(res.content)
# print(res)

#res2 = llm_with_tool.invoke("hello- 'why are you'")
# print(res2.content)
#print(res2)

#checking if the tool_call is faulty or not
# if res2.tool_calls:
#     tool_call = res2.tool_calls[0]

#     #extracting tool name and arguments
#     tool_name= tool_call["name"]
#     tool_args= tool_call["args"]

#     tool_result= get_text_length.invoke(tool_args)

#     final_res = llm_with_tool.invoke(f"Length of text is {tool_result}")

#     print(final_res)

message = []
prompt= input("You : ")
query = HumanMessage(prompt)
message.append(query)
print(message)

result=llm_with_tool.invoke(message)
message.append(result)

if result.tool_calls:
    tool_name= result.tool_calls[0]["name"]# it wont directly work bc too_name is giving a string and we need is function call
    tool_msg= tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_msg)
    #print(message)

res= llm_with_tool.invoke(message)
print(res.content)