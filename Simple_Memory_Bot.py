import os
from dotenv import load_dotenv
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# add your API key to the .env file
# GEMINI_API_KEY=your_api_key_here
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the LLM with the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_output_tokens=1024,
    api_key=api_key
)

# Define the state of the agent
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
def process(state: AgentState) -> AgentState:
    """Solve the request you input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nGEMINI: {response.content}")
    #print("Current STATE:", state["messages"])
    return state

# Create the state graph for the agent
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("You: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    recent_history = conversation_history[-10:]
    result = agent.invoke({"messages": recent_history})
    #print(result["messages"])
    conversation_history.append(result["messages"][-1])
    user_input = input("You: ")

# store memory into text file
with open("memory.txt", "w") as f:
    f.write("Conversation History:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"GEMINI: {message.content}\n")
    f.write("End of conversation.\n")
print("Conversation history saved to memory.txt")