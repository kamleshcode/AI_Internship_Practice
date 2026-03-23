from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

base_url = os.getenv("BASE_URL")

loader = TextLoader("data.txt")
document = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 50,
    chunk_overlap= 20
)

splits = splitter.split_documents(document)

print("Total chunk in document: ", len(splits))

for i,chunks in enumerate(splits):
    print(f"Chunk: {i+1}")
    print(chunks.page_content)
    print(f"Metadata: {chunks.metadata}\n")

context_text = "\n".join([doc.page_content for doc in splits])

template = """
Answer the question based strictly on the provided context. 
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3,
    base_url=base_url,
)

chain = prompt | llm

question = "What is Langchain?"
result = chain.invoke({
    "context": context_text,
    "question": question
})

print(f"Data Extraction Result :\n{result.content}")

# Note:
# Small Files : You join all splits. This is simple but fails if the file is massive (like a 500-page manual).
# Big Files (VectorDB way): You use Similarity Search. It compares your question to all splits and pulls only the top k
# matches (e.g., k=5).
# search_kwargs: This is where you set the limit. If you set k=1, it only sends the single best-matching paragraph to
# the LLM, saving a lot of "space" in the prompt.

# Eg:
# embeddings = OllamaEmbeddings(model="llama3.2")
# vector_db = Chroma.from_documents(documents=splits, embedding=embeddings)
# retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# question = "Who is the Developer and where do they live?"
# docs = retriever.invoke(question)

# context_text = "\n\n".join([doc.page_content for doc in docs])
# print(f"Retrieved {len(docs)} relevant chunks out of {len(splits)} total.")