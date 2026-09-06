from dotenv import load_dotenv

load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct,,,",
    temperature=0.7,
    max_length=1024,
)
model = ChatHuggingFace(llm=llm)
# don't use huggingface via api key as it will bur expensive tokes rather
# use it locally