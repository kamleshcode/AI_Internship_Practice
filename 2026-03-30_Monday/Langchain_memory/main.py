import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    temperature=0.7,
)

memory = InMemorySaver()

session_id = "user_session_1"
mongo_memory = MongoDBChatMessageHistory(
    session_id=session_id,
    connection_string=os.getenv("MONGO_CONN_STRING"),
    database_name = "Langchain",
    collection_name="history"
)

agent = create_agent(
    model=model,
    checkpointer=memory,
)

def trim_and_store(messages):
    if len(messages) >8:
        old_msg = messages[:-8] # take all messages except the last 8
        mongo_memory.add_messages(old_msg)
        return messages[-8:] # keep last 8messages in short-term-memory
    return messages


def search_short_term(query, messages):
    for msg in reversed(messages):
        if query.lower() in msg.content.lower():
            return msg # return matching message content
        return None # if no match found

def search_long_term(query):
    for msg in reversed(mongo_memory.messages):
        if query.lower() in msg.content.lower():
            return msg
        return None


def main():
    print("Simple RAG that can retrieve context")

    local_messages = []

    while True:
        query = input("You : ")

        if query.lower() == "quit":
            break
        short_term = search_short_term(query, local_messages)

        if short_term:
            retrieved_context = short_term
        else:
            retrieved_context = search_long_term(query)


        prompt_text = f"""
        You are a professional AI assistant. 
        
        CONTEXTUAL KNOWLEDGE:
        The following information has been retrieved from long-term memory:
        {retrieved_context if retrieved_context else "No specific background found for this query."}

        INSTRUCTIONS:
        - Provide concise, natural, and helpful responses.
        - Use the 'CONTEXTUAL KNOWLEDGE' to answer accurately if relevant.
        - Never mention phrases like "According to the retrieved memory" or "Based on the context". 
        - Speak like a knowledgeable human colleague.
        """

        system_message = HumanMessage(content=prompt_text)

        result = agent.invoke(
            {"messages": [system_message] + local_messages + [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": "thread-1"}}
        )

        response = result["messages"][-1].content
        print("AI :",response)

        local_messages.append(HumanMessage(content=query))
        local_messages.append(AIMessage(content=response))

        local_messages = trim_and_store(local_messages)

if __name__ == "__main__":
    main()