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

llm = ChatOllama( model=model, base_url=base_url )

# Get the prompt to use - you can modify this!
# Find other prompts to pull here: https://smith.langchain.com/hub?organizationId=338641a2-6da8-4c94-9f15-1fcb10db947b
prompt = hub.pull("hwchase17/react-chat")

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

agent = create_react_agent(llm, tools, prompt)

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
            "tools": tools,
        }
    )
    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=response.get("output")))
    return response.get("output")


#######################################
### Setup Chat interface in console ###
#######################################

# # Now add a simple chat interface
system_prompt = """
You are Ollama Research Bot.
Only use a tool if you need to research something or respond to a question for information.
If given a simple prompt, such as hi, simple respond with your identity and suggest the user to ask a question.
Respond in Markdown format.
"""
def run_chat():
    console = Console()
    chat_history = [SystemMessage(content=system_prompt)]

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Exiting chat...")
                break
            response = send_message(user_input, chat_history)
            console.print(Markdown(response))
        except Exception as e:
            console.print(f"Error occurred: {e}")

# Execute the chat interface
run_chat()