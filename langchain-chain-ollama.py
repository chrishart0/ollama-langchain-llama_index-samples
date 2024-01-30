# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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


llm = ChatOllama( model=model, base_url=base_url )
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# https://python.langchain.com/docs/expression_language/why
chain = prompt | llm | StrOutputParser()

print(chain.invoke( { "topic": "AI" } ))
