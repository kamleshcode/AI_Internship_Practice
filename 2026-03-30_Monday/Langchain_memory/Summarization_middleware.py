import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBChatMessageHistory

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_CONN_STRING")


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

session_id = "user_session_1"
mongo_memory = MongoDBChatMessageHistory(
    connection_string=MONGO_URI,
    session_id=session_id,
    database_name="Langchain",
    collection_name="history"
)

SYSTEM_PROMPT = """
            You are a professional AI assistant that keeps track of the conversation. 
            Guidelines:
            1. Maintain context using short-term memory (last 4 exchanges fully visible).
            2. Retrieve older messages from long-term memory (MongoDB) when needed.
            3. Answer clearly and naturally, human-like.
            4. Recall previous questions and answers accurately. For example:
               User: What is 2+2?
               AI: 2+2 equals 4.
               User: What was my first question?
               AI: Your first question was "What is 2+2?", and the answer was "4".
            5. If an answer is not found in any memory, respond politely indicating lack of information.
            """
agent = create_agent(
    model=llm,
    checkpointer=None,
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=("messages", 4),  # summarize when >4 messages
            keep=("messages", 4),     # keep last 4 messages in full
        )
    ]
)

def update_long_term(local_messages):
    if len(local_messages) > 8:  # more than 4 user + 4 AI messages
        # store oldest messages to MongoDB
        old = local_messages[:-8]
        for msg in old:
            if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
                mongo_memory.add_message(msg)
        return local_messages[-8:]
    return local_messages

def search_long_term(query):
    for msg in reversed(mongo_memory.messages):
        if query.lower() in msg.content.lower():
            return msg
    return None


def chat():
    thread_id = "rag_thread_1"
    local_messages = []

    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        retrieved_msg = search_long_term(query)
        content = query
        if retrieved_msg:
            content += f"\n(Context from older memory: {retrieved_msg.content})"

        prompt_message = HumanMessage(content=content)

        result = agent.invoke(
            {"messages": [prompt_message]},
            config={"configurable": {"thread_id": thread_id}}
        )

        response_msg = result["messages"][-1]
        print("\nAI:", response_msg.content)


        local_messages.append(HumanMessage(content=query))
        local_messages.append(response_msg)

        local_messages = update_long_term(local_messages)


if __name__ == "__main__":
    chat()