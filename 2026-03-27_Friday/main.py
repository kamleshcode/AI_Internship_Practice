import asyncio
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
)

agent = create_agent(
    model=llm,
)

#Invoke
response = agent.invoke({"messages": [("user", "What is the weather in Paris?")]})
print(response["messages"][-1].content)

# Streaming
stream = agent.stream({"messages": [("user", "Explain state in LangChain in 20 words")]})
print(stream)
for chunk in stream:
    if "agent" in chunk:
        print(chunk["agent"]["messages"][-1].content)

# Batch
questions = [
    {"messages": [("user", "What is a transformer in AI?")]},
    {"messages": [("user", "What is 2+2?")]},
    {"messages": [("user", "Difference between stream and batch in LangChain?")]},
]

response3 = agent.batch(
    questions,
    config={"max_concurrency": 2}
)

for response in response3:
    print(response)

# ainvoke
async def run_ainvoke():
    print("\nRunning ainvoke..")
    res = await agent.ainvoke({"messages": [("user", "Explain LangChain in 20 words")]})
    print(res["messages"][-1].content)

# astream
async def run_astream():
    print("\nRunning astream..")
    async for chunk in agent.astream({"messages": [("user", "What is 2+2?")]}):
        if "agent" in chunk:
            print(f"Stream: {chunk['agent']['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(run_ainvoke())
    asyncio.run(run_astream())