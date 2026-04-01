from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import os

llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

def load_pdf_text(path:str):
    """Load and extract complete text from PDF file."""
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        return full_text
    except Exception as e:
        print(f"Error in loading PDF: {e}")
        return ""

def load_prompt_template(path:str):
    """Load markdown prompt file and return as a PromptTemplate."""
    try:
        # Use standard file reading for Markdown (.md) files
        with open(path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        return PromptTemplate.from_template(prompt_content)
    except Exception as e:
        print(f"Error in loading Prompts: {e}")
        return None

@tool
def generate_requirements(document_text:str):
    """Generate detailed functional and non-functional requirements strictly from provided business document content."""
    try:
        print("[Calling tool : generate_requirements]")
        prompt = load_prompt_template("prompts/requirement_prompt.md")
        formated_prompt = prompt.format(context=document_text)
        requirements = llm.invoke(formated_prompt).content
        print("\nOutput of generate_requirements :\n", requirements)
        return requirements
    except Exception as e:
        print(f"Error in generate_requirements: {e}")

@tool
def generate_user_stories(requirements:str):
    """
    Generate detailed user stories from requirements.Include title,description, and detailed acceptance criteria
    covering positive and negative flows.
    """
    try:
        print("[Calling tool : generate_user_stories]")
        prompt = load_prompt_template("prompts/user_story_prompt.md")
        formated_prompt=prompt.format(requirements=requirements)
        user_stories = llm.invoke(formated_prompt).content
        print("\nOutput of generate_user_stories :\n", user_stories)
        return user_stories
    except Exception as e:
        print(f"Error in generate_user_stories: {e}")

@tool
def generate_tasks(stories:str):
    """Generate implementation tasks from user stories. One user story may generate multiple tasks."""
    try:
        print("[Calling tool : generate_tasks]")
        prompt = load_prompt_template("prompts/task_prompt.md")
        formated_prompt=prompt.format(stories=stories)
        tasks = llm.invoke(formated_prompt).content
        print("\nOutput of generate_tasks :\n", tasks)
        return tasks
    except Exception as e:
        print(f"Error in generate_tasks: {e}")

