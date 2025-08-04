import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from operator import add as add_messages

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_output_tokens=1024,
    api_key=api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"    
)

pdf_path = "CSE.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try: 
    pages = pdf_loader.load()
    print(f"Loaded {len(pages)} pages from the PDF.")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# chunking the text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
pages_split = text_splitter.split_documents(pages)

persist_directory = "chroma_db"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    #create the chroma vector store
    vactorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name 
    )

    print(f"Vector store is created. Total chunks: {len(pages_split)}")

except Exception as e:
    print(f"Error creating vector store: {e}")
    raise

#retriever
returner = vactorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from Colonbo Stock market Exchange documents."""
    docs =  retriever.invoke(query)

    if not docs:
        return "I found no relevant information about Colombo Stock Market Exchange."
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i}: {doc.page_content}")
    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls"""
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_promt = """
You are an intelligent AI assistant who answers questions about Colombo Stock Market. Use 
retriever tool available to answer questions about the Colombo Stock Market performance 
data. If you need to look up some information before asking a follow up question, you are 
allowed to do research on the given data. Please always cite the specific parts of the 
documents you use in your answers.
"""

tools_dict = { our_tool.name : our_tool for our_tool in tools}

def call_llm(state: AgentState) -> AgentState:
    """Fuction to call the LLM with current state"""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_promt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response"""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling tool: {t['name']} with query: {t['args'].get('query','no query provided')}")
    
        if not t['name'] in tools_dict: 
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Tools execution complete. Back to the model")    
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriver_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriver_agent", False: END}
)
graph.add_edge("retriver_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG Agent ===")

    while True:
        user_input = input("\nWhat is your question : ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        
        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("\n ======= ANSWER =======")
        print(result['messages'][-1].content)
    
running_agent()  

