from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

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
    prompt = load_prompt_template("prompts/requirement_prompt.md")
    if prompt:
        return prompt.format(context=document_text)
    return "Error: Could not load requirement prompt."

@tool
def generate_user_stories(requirements:str):
    """
    Generate detailed user stories from requirements.Include title,description, and detailed acceptance criteria
    covering positive and negative flows.
    """
    prompt = load_prompt_template("prompts/user_story_prompt.md")
    if prompt:
        return prompt.format(requirements=requirements)
    return "Error: Could not load user story prompt."

@tool
def generate_tasks(stories:str):
    """Generate implementation tasks from user stories. One user story may generate multiple tasks."""
    prompt = load_prompt_template("prompts/task_prompt.md")
    if prompt:
        return prompt.format(stories=stories)
    return "Error: Could not load task prompt."
