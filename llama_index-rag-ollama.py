# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from llama_index.llms import Ollama
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.callbacks import CallbackManager

### Configs ###
# Load model configurations from .env file
# Nice to keep configs in one place to ensure model stays same across files. 
# Changing model takes a long time for first load
from dotenv import load_dotenv
import os
load_dotenv()
model=os.getenv("MODEL")
base_url=os.getenv("BASE_OLLAMA_URL")
################

### Monitoring Setup ###
import phoenix as px

# To view traces in Phoenix, you will first have to start a Phoenix server. You can do this by running the following:
session = px.launch_app()

from phoenix.trace.llama_index import (
    OpenInferenceTraceCallbackHandler,
)
#########################

# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(
    model=model,
    base_url=base_url
)
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

### Llama_index Detailed logging ###
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
####################################

# Initialize the callback handler for tracing
callback_handler = OpenInferenceTraceCallbackHandler()

# Load data
reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()

# Initialize QdrantClient and QdrantVectorStore
qdrant_client = QdrantClient(":memory:") # QdrantClient(path="./qdrant_data")

qdrant_vector_store = QdrantVectorStore(
    qdrant_client=qdrant_client, 
    collection_name="zarathustra",
    client = qdrant_client
)

qdrant_storage_context = StorageContext.from_defaults(
    vector_store=qdrant_vector_store
)

# Initialize Ollama and ServiceContext
print("Initializing Ollama...")
llm = Ollama(
    model="mixtral",
    base_url="http://0.0.0.0:11435",
    request_timeout=100
)
service_context = ServiceContext.from_defaults(
    llm=llm, 
    embed_model="local",
    callback_manager=CallbackManager(handlers=[callback_handler]),
)

# Create VectorStoreIndex and query engine
print("Creating index...")
qdrant_index = VectorStoreIndex.from_documents(
    documents, storage_context=qdrant_storage_context, service_context=service_context
)

print("Creating query engine...")
query_engine = qdrant_index.as_query_engine()

# Perform a query and print the response
print("Querying...")

# Create a python chat which takes users input and returns a response in a loop
print("Starting chat...")
while True:
    user_input = input("You: ")
    # If the user types "exit", exit the loop
    if user_input == "exit":
        break
    response = query_engine.query(user_input)
    print(response)

px.active_session().url