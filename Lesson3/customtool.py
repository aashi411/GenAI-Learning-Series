from langchain.tools import tool
#to convert the def into a tool by making it a decorator
#tools are also known as runnables

@tool
def get_greeting(name: str) -> str:
    """Generate a greeting message - this string is called docString"""# docstring-is the official description 
    return f"Hello {name} welcome to the ai world"

res= get_greeting.invoke({"name": "Aashi"})
print(res)
print(get_greeting.description)
print(get_greeting.name)
print(get_greeting.args)


