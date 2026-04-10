import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.mongodb import MongoDBSaver

MONGO_URI = os.getenv("MONGO_CONN_STRING")

client = MongoClient(MONGO_URI)
collection = client["chat_db"]["history"]

memory = MongoDBSaver(client = client, collection = "histroy")
