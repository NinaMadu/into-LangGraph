import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage,ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# add your API key to the .env file
# GEMINI_API_KEY=your_api_key_here
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

document_content = ""

# Define the state of the agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Initialize the LLM with the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_output_tokens=1024,
    api_key=api_key
)

@tool
def update(content: str) -> str:
    """Update the document content with the provided text."""
    global document_content
    document_content = content
    return f"Document has been updated successfully. The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document content to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """

    if not filename.endswith(".txt"):
        filename += ".txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"Document saved successfully to {filename}.")    
        return f"Document saved successfully to {filename}."
    except Exception as e:        
        return f"Error saving document: {str(e)}"
    
tools = [update, save]

model = llm.bind_tools(tools)

def our_agent(state:AgentState) -> AgentState:
    """Solve the request you input"""
    system_prompt = SystemMessage(content=f"""
        You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

        -If the user wants to update or modify the content, use the 'update' tool with complete updated content.
        -If the user wants to save and finish, you need to use the 'save' tool.
        -Make sure to always show the current document state after modifications.

        The current document content is: {document_content}
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\nUser : {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response =model.invoke(all_messages)

    print(f"\n AI : {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\nUsing tools : {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState):
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]

    if not messages:
        return "continue"    
    
    for message in reversed(messages):
        #check if it is a tool message after Saveing 
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return END
        
    return "continue"

def print_messages(messages):
    """Function to print messages in more readable format."""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n Tool Result: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n +++++++++++++++++ DRAFTER +++++++++++++++++ \n")
    state = { "messages": [] }

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n +++++++++++++++++ END OF DRAFTER +++++++++++++++++ \n")

if __name__ == "__main__":
    run_document_agent()
