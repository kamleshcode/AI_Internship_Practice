from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from rag import data_retriever
import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.memory import MemorySaver

def main():
    try:
        file_paths = ["data/company_policies.pdf","data/company_policies.pdf","data/product_manual.pdf"]
        retriever = data_retriever(file_paths)

        @tool
        def search_document(query):
            """this is used to retrieve data from the retriever"""
            docs = retriever.invoke({"query": query})
            print(docs)
            return "\n".join([d.content for d in docs])

        llm = ChatOllama(
            model="mistral-nemo",
            temperature =0.7,
            base_url=os.getenv("BASE_URL"),
        )

        tools = [search_document]

        agent = create_agent(
            model= llm,
            tools = tools,
            middleware=[SummarizationMiddleware(
                model="mistral-nemo",
                trigger =("messages", 6),
                keep = ("messages",6)
            )],
            checkpointer = MemorySaver(),
        )

        while True:
            print("Started")
            query = input("Enter query: ")
            if query.lower() == "exit":
                exit()
            docs = retriever.invoke({"query": query})
            print(docs)
            config = {"configurable": {"thread_id": "thread_1"}}
            response = agent.invoke({"messages":[{"role": "user", "content": query}]}, config=config)
            print(response)
            content = response["messages"][-1].content
            print(f'AI Response: {content}')

    except Exception as e:
        print(f'Error in main function: {e}')

if __name__ == "__main__":
    main()
