import os
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

@tool
def get_stock_price(ticker: str) -> str:
    """Fetches the current market price for a stock ticker."""
    return f"The current price of {ticker} is Rs.150.00"

@tool
def execute_trade(ticker: str, action: str, quantity: int) -> str:
    """Executes a buy or sell order for a specific stock."""
    return f"Successfully {action}ed {quantity} shares of {ticker}."

tools = [get_stock_price, execute_trade]

# NOTE :Human-in-the-loop middleware requires a checkpointer to maintain state across interruptions.
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
    You are a financial assistant. To use a tool, you MUST return a tool call. 
    Do NOT nest parameters inside a 'parameters' or 'function' key.
    Example: If calling get_stock_price, provide: {'ticker': 'AAPL'}.
    Current tools available: get_stock_price, execute_trade.
    """,
    checkpointer=InMemorySaver(),
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            "get_stock_price": {"allowed_decisions": ["approve", "reject"]},
            "execute_trade": False,
        }
    )],
)


def main():
    config = {"configurable": {"thread_id": "trading_thread_1"}}
    query = {"messages": [HumanMessage(content="Check the price of AAPL and buy 5 shares if it is Rs.150.")]}

    print("Step 1: Initial Invocation (Should Interrupt)")
    result = agent.invoke(query, config=config)

    if "messages" in result:
        print(f"Agent Action: {result['messages'][-1].additional_kwargs.get('tool_calls', 'No tool call found')}")

    print("\nStep 2: Sending Approval Command")
    final_state = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "approve"
                    }
                ]
            }
        ),
        config=config
    )

    print("\nFinal Agent Response")
    if "messages" in final_state:
        print(final_state["messages"][-1].content)
    else:
        print("Error: No messages found in final state.")


if __name__ == "__main__":
    main()



