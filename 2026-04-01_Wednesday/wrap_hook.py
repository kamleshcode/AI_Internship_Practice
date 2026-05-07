import os
import time
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
)

@wrap_model_call
def timing_middleware(request, handler):
    print("\nBefore model call ")
    start = time.time()
    # handler returns a ModelResponse object
    response = handler(request)
    print(response)
    end = time.time()
    print("After model call")
    print(f"Execution time: {end - start:.2f} sec")
    if response.result and len(response.result) > 0:
        response.result[0].content += "\n\n[Processed through middleware]"

    return response

agent = create_agent(
    model=model,
    tools=[],
    middleware=[timing_middleware]
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain what is machine learning in simple words"
            }
        ]
    }
)

for msg in result["messages"]:
    print(f"\n{msg.type.upper()} : {msg.content}")