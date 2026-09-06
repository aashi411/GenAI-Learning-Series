from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
model = init_chat_model("google_genai:gemini-3.5-flash")

response= model.invoke("How to become a DJ?- explain in a paragraph")
print(response.content)