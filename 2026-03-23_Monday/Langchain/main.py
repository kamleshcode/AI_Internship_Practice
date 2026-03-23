import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()

base_url = os.getenv("BASE_URL")

llm = ChatOllama(
    model = "llama3.2",
    temperature = 0.3,
    base_url = base_url,
)

prompt = ChatPromptTemplate.from_template("Tell me a funny joke about {topic}.")

chain = prompt | llm
result = chain.invoke({"topic": "programming"})
print(result.content)

