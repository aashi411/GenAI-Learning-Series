from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
model = ChatMistralAI(model= "mistral-small-2506", temperature=0.9)

print("Choose your AI model")
print("1- angry mode")
print("2- funny mode")
print("3- gloomy mode")

choice = int(input("Enter your choice: "))

#short term memory
if choice ==1:
    mode = "You are an angry Ai agent, response must be aggressive and impatient."
elif choice==2:
    mode = "You are an funny Ai agent, response must be unserious and comical."
elif choice==3:
    mode = "You are an gloomy Ai agent, response must be sad and depressing."


messages = [
    SystemMessage(content= mode)
]
print("_________________________Welcome (Type 0 to exit)_______________________________)")

while(True):
    prompt= input("You: ")
    messages.append(HumanMessage(content=prompt))
    if(prompt == "0"):
        break
    response= model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot: ", response.content)
print(messages)