import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient

load_dotenv()

# Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Tool
@tool("web_search", description="Search latest web information.")
def internet_search(user_query: str):
    return tavily_client.search(user_query)

# Model builder
def get_model(model_name, temp):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=model_name,
        temperature=temp
    )
    return llm.bind_tools([internet_search])

# Response generator
def get_response(messages, model_name, temp):
    model = get_model(model_name, temp)

    response = model.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            try:
                # Execute tool via .invoke for proper mapping
                search_results = internet_search.invoke(tool_call["args"])
                messages.append(ToolMessage(
                    content=str(search_results),
                    tool_call_id=tool_call["id"]
                ))
            except Exception as tool_err:
                messages.append(ToolMessage(
                    content=f"Tool Error: {str(tool_err)}",
                    tool_call_id=tool_call["id"]
                ))

        final_response = model.invoke(messages)
        messages.append(final_response)

        return final_response.content

    return response.content

