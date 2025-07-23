import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# add your API key to the .env file
# GEMINI_API_KEY=your_api_key_here
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Define the state of the agent
class AgentState(TypedDict):
    messages: List[HumanMessage]

# Initialize the LLM with the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_output_tokens=1024,
    api_key=api_key
)

# Define the process function that will handle the conversation
def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nGEMINI: {response.content}")
    return state

# Create the state graph for the agent
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Start the conversation with the user
print("Welcome to the Gemini Simple Chatbot!")
print("\nType 'exit' to end the conversation.\n")
user_input = input("You: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("You: ")
print("Goodbye!")

    