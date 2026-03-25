import os
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
    pdf_path = "data/SRS.pdf"

    document_text = load_pdf_text(pdf_path)

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

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
    You are a Business Analyst Agent.
    
    Workflow:
    1. Generate requirements
    2. Generate user stories
    3. Generate tasks
    
    Rules:
    - Use only document content
    - Do not invent missing business logic
    """
    )

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
                Document Content:
                {document_text}

                Generate functional requirements, non-functional requirements,
                detailed user stories, and implementation tasks.
                """
            },
        ]
    })


    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
