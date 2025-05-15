from typing import TypedDict, Annotated, List
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
import chainlit as cl
import operator

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document



load_dotenv()

##-----------------------------------------------------------------------------------
#                       reading data
##-----------------------------------------------------------------------------------

path = "Data/"
loader = DirectoryLoader(path, glob="*.html")
docs = loader.load()



##-----------------------------------------------------------------------------------
#                       OTHER TOOLS
##-----------------------------------------------------------------------------------
tavily_tool = TavilySearchResults(max_results=5)
arxiv_tool = ArxivQueryRun()




##-----------------------------------------------------------------------------------
#                       R - PREPATION OF THE GRAPH RAG
##-----------------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
split_documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="obesity_challange",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="obesity_challange",
    embedding=embeddings,
)

_ = vector_store.add_documents(documents=split_documents)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def retrieve_adjusted(state):
    compressor = CohereRerank(model="rerank-v3.5", top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever, search_kwargs={"k": 5}
    )
    retrieved_docs = compression_retriever.invoke(state["question"])
    return {"context": retrieved_docs}


RAG_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.
### Question
{question}
### Context
{context}
"""



##-----------------------------------------------------------------------------------
#                       G - PREPARATION OF GRAPH RAG
##-----------------------------------------------------------------------------------

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

llm = ChatOpenAI(model="gpt-4o-mini")

def generate(state):
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
  response = llm.invoke(messages)
  return {"response" : response.content}


##-----------------------------------------------------------------------------------
#                       GRAPH RAG 
##-----------------------------------------------------------------------------------


class State(TypedDict):
  question: str
  context: List[Document]
  response: str

graph_rag_builder = StateGraph(State).add_sequence([retrieve_adjusted, generate])
graph_rag_builder.add_edge(START, "retrieve_adjusted")
graph_rag = graph_rag_builder.compile()


##-----------------------------------------------------------------------------------
#                       TOOLS PREPATION FOR AGENT
##-----------------------------------------------------------------------------------

@tool
def obesity_rag_tool(question: str) -> str:
  """Useful for when you need to answer questions about artificial intelligence. Input should be a fully formed question."""
  response = graph_rag.invoke({"question" : question})
  return {
        "messages": [HumanMessage(content=response["response"])],
        "context": response["context"]
    }

tool_belt = [
    tavily_tool,
    arxiv_tool,
    obesity_rag_tool
]


##-----------------------------------------------------------------------------------
#                       MODELS WITH TOOLS
##-----------------------------------------------------------------------------------

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
model = model.bind_tools(tool_belt)

##-----------------------------------------------------------------------------------
#                       AGENT GRAPH
##-----------------------------------------------------------------------------------

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]
  context: List[Document]

tool_node = ToolNode(tool_belt)

uncompiled_graph = StateGraph(AgentState)

def call_model(state):
  messages = state["messages"]
  response = model.invoke(messages)
  return {
        "messages": [response],
        "context": state.get("context", [])
    }

uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", tool_node)

uncompiled_graph.set_entry_point("agent")

def should_continue(state):
  last_message = state["messages"][-1]

  if last_message.tool_calls:
    return "action"

  return END

uncompiled_graph.add_conditional_edges(
    "agent",
    should_continue
)

uncompiled_graph.add_edge("action", "agent")

compiled_graph = uncompiled_graph.compile()

#------------------------------------------------------------------------

# @cl.on_chat_start
# async def start():
#   cl.user_session.set("graph", compiled_graph)
#   await cl.Message(content="Hello! I'm ready to help with your questions.").send()

# @cl.on_message
# async def handle(message: cl.Message):
#   graph = cl.user_session.get("graph")
#   state = {"messages" : [HumanMessage(content=message.content)]}
#   response = await graph.ainvoke(state)
#   await cl.Message(content=response["messages"][-1].content).send()

@cl.on_chat_start
async def start():
    # Initialize with the compiled graph
    cl.user_session.set("graph", compiled_graph)
    
    # Initialize an empty state with the structure expected by your graph
    initial_state = {"messages": [], "context": []}
    cl.user_session.set("state", initial_state)
    
    # Send a welcome message to the UI
    welcome_message = """
# 👋 Hello! I am a specialized assistant focused on obesity research and health information.

I'm designed to provide evidence-based information from trusted sources including:
- 📚 NIH Director's Blog on obesity research
- 🔬 Scientific definitions and classifications
- 📊 Data-driven insights about health impacts
- 🩺 Information about treatment approaches

**My goal is to provide accurate, non-judgmental information about obesity as a health condition.**

How can I assist with your obesity-related questions today?
    """
    
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def handle(message: cl.Message):
    # Show typing indicator
    thinking = cl.Message(content="Thinking...")
    await thinking.send()
    
    try:
        # Get the graph and current state
        graph = cl.user_session.get("graph")
        current_state = cl.user_session.get("state", {"messages": [], "context": []})
        
        # Add the new user message to the existing messages
        updated_messages = current_state["messages"] + [HumanMessage(content=message.content)]
        
        # Create an updated state
        updated_state = {
            "messages": updated_messages,
            "context": current_state.get("context", [])
        }
        
        # Invoke the graph with the updated state
        response = await graph.ainvoke(updated_state)
        
        # Store the updated state for the next interaction
        cl.user_session.set("state", response)
        
        # Remove the typing indicator
        await thinking.remove()
        
        # Get the latest message (the AI's response)
        if response["messages"] and len(response["messages"]) > len(updated_messages):
            ai_message = response["messages"][-1]
            await cl.Message(content=ai_message.content).send()
        else:
            # Fallback if no new message was added
            await cl.Message(content="I'm sorry, I couldn't generate a response.").send()
            
    except Exception as e:
        # Handle any errors
        error_message = f"Error processing your request: {str(e)}"
        await thinking.update(content=error_message)
        print(f"Error: {str(e)}")
