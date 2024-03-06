####################################################################################
# Import Libraries
####################################################################################
from dotenv import load_dotenv
import logging
import os
import phoenix as px
import sys
from qdrant_client import QdrantClient
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from llama_index.callbacks import CallbackManager
from llama_index.llms import Ollama
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler

####################################################################################
# Configuration Setup
####################################################################################
# Load environment variables for consistent configuration across the application.
load_dotenv()
model = os.getenv("MODEL")
base_url = os.getenv("BASE_OLLAMA_URL")

# Configure logging for detailed output. Adjust or disable for less verbosity.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

####################################################################################
# Monitoring Setup with Phoenix
####################################################################################
# Initialize Phoenix monitoring. Ensure a Phoenix server is running beforehand.
session = px.launch_app()

####################################################################################
# Initialize Ollama Model
####################################################################################
# Setup Ollama model with specific configurations for use in chat interactions.
print("Initializing Ollama...")
llm = ChatOllama(
    model=model,
    base_url=base_url,
    request_timeout=100  # Adjust timeout for model responsiveness.
)

# Prepare a template for generating chat prompts.
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

####################################################################################
# Callback Handler Setup
####################################################################################
# Setup callback handler for tracing interactions within Phoenix.
callback_handler = OpenInferenceTraceCallbackHandler()

####################################################################################
# Data Preparation
####################################################################################
# Load documents from specified directory for indexing and querying.
reader = SimpleDirectoryReader(input_dir="./data/ben-franklin")
documents = reader.load_data()

####################################################################################
# Qdrant Client and Vector Store Setup
####################################################################################
# Initialize QdrantClient and Vector Store for document vector management.
qdrant_client = QdrantClient(":memory:")  # In-memory storage for examples.
# Alternative storage options:
# qdrant_client = QdrantClient(path="./qdrant_data")  # Local filesystem storage.
# qdrant_client = QdrantClient(url="http://localhost:6333")  # Qdrant container setup.

qdrant_vector_store = QdrantVectorStore(
    qdrant_client=qdrant_client,
    collection_name="ben_franklin",
    client = qdrant_client
)

qdrant_storage_context = StorageContext.from_defaults(
    vector_store=qdrant_vector_store
)

####################################################################################
# Service Context and Index Initialization
####################################################################################
# Configure the service context for use with LlamaIndex and local language model.
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local",
    callback_manager=CallbackManager(handlers=[callback_handler]),
)

# Create a VectorStoreIndex and query engine based on the loaded documents.
print("Creating index...")
qdrant_index = VectorStoreIndex.from_documents(
    documents, storage_context=qdrant_storage_context, service_context=service_context
)

print("Creating query engine...")
query_engine = qdrant_index.as_query_engine()

####################################################################################
# User Interaction Loop
####################################################################################
# Interactive chat loop for querying the engine with user input.
print("") # New line for readability
print("Note: This is a pure LlamaIndex RAG chat. It will always try to do a RAG query, so simple statements like 'hi' will produce an unrelated RAG response.")
print("Starting chat...")
print("Try a question like: 'Where was Ben Franlkin Born?' or 'When did Ben Franklin move to Philidelphia?'")
print ("Type 'exit' to end the chat.")
print("") # New line for readability
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    response = query_engine.query(user_input)
    print(response)
    print("")# New line for readability

####################################################################################
# Phoenix Monitoring URL
####################################################################################
# Display the URL for active session monitoring in Phoenix.
print("Phoenix Monitoring URL:", px.active_session().url)
