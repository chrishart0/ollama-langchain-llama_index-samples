# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain import hub
from langchain.agents import tool, create_react_agent
from langchain.tools import Tool
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Duck Duck Go API search
from duckduckgo_search import DDGS

# Make the chat look nice
from rich.console import Console
from rich.markdown import Markdown

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


### Monitoring and Tracing for LangChain ###
import phoenix as px
from phoenix.trace.langchain import OpenInferenceTracer, LangChainInstrumentor

# To view traces in Phoenix, you will first have to start a Phoenix server. You can do this by running the following:
session = px.launch_app()

tracer = OpenInferenceTracer()
LangChainInstrumentor(tracer).instrument()
#############################

llm = ChatOllama(  model=model, base_url=base_url )

# Get the prompt to use - you can modify this!
# prompt = hub.pull("archit0/react-chat-json")
prompt = hub.pull("hwchase17/react")
# prompt = hub.pull("homanp/superagent") # Give non openai models open-ai like ability to use tools

### Tools Setup ###
@tool
def search_duckduckgo_langchain_tool(query: str) -> int:
    """Takes a search query and search for Duck Duck Go results"""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return results

tools = [
    Tool(
        name="Search", # Model seems to 
        description="Search the internet for results using Duck Duck Go",
        func=search_duckduckgo_langchain_tool.run,
    )
]

######################

# agent = create_openai_functions_agent(llm, tools, prompt)
agent = create_react_agent(
    llm, 
    tools, 
    prompt
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True # Give tool errors back to the LLM so it can deal with them as it sees fit
)


def send_message(message, chat_history):
    response = agent_executor.invoke(
        {
            "input": message,
            "chat_history": chat_history,
            "tool_names": ["Search"],
            "tools": tools,
        }
    )

    return response.get("output")


#######################################
### Setup Chat interface in console ###
#######################################

# # Now add a simple chat interface
system_prompt = """
You are OllamaBot. 
"""
def run_chat():
    console = Console()
    chat_history = [ SystemMessage (content=system_prompt)]

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chat...")
            break
        response = send_message(user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        console.print(Markdown(response))

# Execute the chat interface
run_chat()