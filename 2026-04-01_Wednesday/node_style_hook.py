"""
Custom Middleware
- Build custom middleware by implementing hooks
- Middleware provides two styles of hooks to intercept agent execution:
    1.Node-style hooks(run at a specific "point in time." - Use for logging, validation, and state updates.)
        - before_agent
        - before_model
        - after_agent
        - after_model
    2.Wrap-style hooks(used for retries, dynamic model selection, fallback model)
        - in wrap middleware two object arrives(request -contain current model input, handler - contain next
        execution step)
        -wrap_model_call
        -wrap_tool_call

-State (is the live working memory of the agent during one execution)
"""
import os
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.runtime import Runtime
from typing import Any
from dotenv import load_dotenv
load_dotenv()

@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 1:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None

model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    temperature=0
)

agent = create_agent(
    model,
    middleware=[check_message_limit, log_response]
)

inputs = {"messages": [HumanMessage(content="Explain middleware in one sentence.")]}
result = agent.invoke(inputs)

limit_inputs = {"messages": [HumanMessage(content="Hi")] * 50}
limit_result = agent.invoke(limit_inputs)

print(f"\nFinal Agent Response: {limit_result['messages'][-1].content}")