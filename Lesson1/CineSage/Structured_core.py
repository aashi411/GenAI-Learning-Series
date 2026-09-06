#Chat-prompt templates
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

model = ChatMistralAI (model= 'mistral-small-2506')

#class ke ander class(Basemodel) ko inherit kar rahe for pydentic schema
class Movie(BaseModel):
    title: str
    release_yr: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)


# Instantiation using from_template (recommended)
prompt = ChatPromptTemplate.from_messages(
[
       ('system', """ 
        Extract movie information from the paragraph
        {format_instructions}
        """
        ), 
        ("human", "{paragraph}")
]
    
)

para = input("Give your paragraph: ")

final_prompt= prompt.invoke(
    {"paragraph": para,
     "format_instructions": parser.get_format_instructions()}
)

response = model.invoke(final_prompt)

movie = parser.parse(response.content)
print(movie.model_dump_json(indent=4))