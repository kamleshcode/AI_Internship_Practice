import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def data_retriever(file_path):
    """
    This function is used to retrieve the data from the vectorDB
    """
    all_docs=[]
    for path in file_path:
        loader = PyPDFLoader(path)
        data = loader.load()
        all_docs.extend(data)

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks= splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(
        model = os.getenv("OLLAMA_EMBEDDING_MODEL"),
        base_url=os.getenv("BASE_URL"),
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory='./chromadb'
    )
    return vectorstore.as_retriever(search_kwargs={"k": 6})

