from dataclasses import dataclass
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Context:
    user_name: str
    account_tier: str  # e.g., "VIP" or "Free"
    current_order_id: str

@tool
def check_shipping_tool(runtime: ToolRuntime[Context]) -> str:
    """Looks up current shipping status using live runtime context."""
    user = runtime.context.user_name
    order = runtime.context.current_order_id
    msg_count = len(runtime.state.get("messages", []))

    return f"Hi {user}, for order {order} (Turn {msg_count}), your package is out for delivery!"

llm = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0
)
memory_saver = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[check_shipping_tool],
    context_schema=Context,
    checkpointer=memory_saver
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "john_smith_session_001"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Where is my package?"}]},
        config=config,
        context=Context(user_name="John Smith", account_tier="VIP", current_order_id="XYZ-1234")
    )

    print(response)
