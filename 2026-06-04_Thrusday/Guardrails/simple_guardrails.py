import os
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_ADVANCE_MODEL"),
    temperature=0.3
)

agent = create_agent(
    model=llm,
    system_prompt="You are a helpful assistant that provides concise answers to questions.",
    middleware=[
        PIIMiddleware(""
                      "email",
                      strategy="redact",
                      apply_to_input=True
                      ),
        PIIMiddleware("api_key",
                      detector=r"sk-[a-zA-Z0-9]{32}",
                      strategy="block",
                      apply_to_input=True
                      )
    ]
)

messages = {
    "messages" : HumanMessage(content="my email is alok@gmail.com and my api key is skdfjgdnv5fddsvsffbgbaxcv")
}

result = agent.invoke(messages)
print(result["messages"][-1].content)