import os
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

from tools import (
    load_pdf_text,
    generate_requirements,
    generate_user_stories,
    generate_tasks
)

def main():
    pdf_path = "data/sample.pdf"

    try:
        document_text = load_pdf_text(pdf_path)
        print(f"\nPDF Loaded: {len(document_text)} characters.\n")
    except Exception as e:
        print("Failed to load PDF content.", e)
        return

    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    tools = [
        generate_requirements,
        generate_user_stories,
        generate_tasks
    ]

    # Create Agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
    You are a Business Analyst Agent.
    Workflow:
    1. Generate functional and non-functional requirements from the provided document.
    2. Generate detailed user stories based on the requirements.
    3. Generate implementation tasks from the user stories.
    
    Rules:
    - Use only the provided document content.
    - Do not invent missing business logic.
    - Always call tools in order: generate_requirements -> generate_user_stories -> generate_tasks

    """
    )

    # Invoke agent
    print("\nAgent Starting BA Automation..\n")
    user_message = HumanMessage(
        content=f"""
            Document Content: {document_text}

            Generate functional requirements, non-functional requirements,
            detailed user stories, and implementation tasks.
            """
    )

    result = agent.invoke({"messages": [user_message]})

    print("\nAgent Final Output\n")
    for msg in result["messages"]:
        print(msg.type.upper(), ":", msg.content)

if __name__ == "__main__":
    main()
