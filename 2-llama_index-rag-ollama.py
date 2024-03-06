####################################################################################
# Import Libraries
####################################################################################
from dotenv import load_dotenv
import logging
import os
import phoenix as px
import sys
from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
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
print("Configuring Ollama...")
Settings.llm = Ollama(
    model=model,
    base_url=base_url,
    request_timeout=120  # Adjust timeout for model responsiveness.
)
Settings.context_window = 4096

# Must us a local embeddings model, otherwise will default to OpenAI
print("Configuring Local Embeddings Models...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

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

# Create a VectorStoreIndex and query engine based on the loaded documents.
print("Creating index...")
qdrant_index = VectorStoreIndex.from_documents(
    documents, storage_context=qdrant_storage_context
)

print("Creating query engine...")
query_engine = qdrant_index.as_query_engine()


####################################################################################
# Phoenix Monitoring URL
####################################################################################
# Display the URL for active session monitoring in Phoenix.
print("Phoenix Monitoring URL:", px.active_session().url)

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