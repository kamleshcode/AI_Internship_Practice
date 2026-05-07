import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware
)
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain.tools import tool

load_dotenv()

main_model = ChatOllama(
    model=os.getenv("OLLAMA_ADVANCE_MODEL"),
    temperature=0
)

fallback_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def weather(city: str) -> str:
    """Get weather information."""
    return f"Weather in {city}: 32°C, clear sky"


tools = [calculator, weather]

memory = InMemorySaver()

agent = create_agent(
    model=main_model,
    tools=tools,
    checkpointer=memory,

    middleware=[

        SummarizationMiddleware(
            model=main_model,
            trigger=("messages", 8),
            keep=("messages", 4),
        ),

        ToolRetryMiddleware(
            max_retries=2,
            initial_delay=1,
            backoff_factor=2
        ),

        ModelRetryMiddleware(
            max_retries=2,
            initial_delay=1,
            backoff_factor=2
        ),

        ToolCallLimitMiddleware(
            thread_limit=20,
            run_limit=5
        ),

        ModelCallLimitMiddleware(
            thread_limit=30,
            run_limit=5,
            exit_behavior="end"
        ),

        ModelFallbackMiddleware(
            fallback_model
        )
    ]
)

config = {
    "configurable": {
        "thread_id": "user-session-001"
    }
}

while True:
    query = input("\nYou: ")

    if query.lower() == "exit":
        break

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": query}
            ]
        },
        config=config
    )

    print("\nAgent:", result["messages"][-1].content)