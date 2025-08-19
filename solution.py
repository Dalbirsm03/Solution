from langgraph.graph import END , START , StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image , display
from typing import TypedDict , Literal
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant")

class State(TypedDict):
    input: str
    choice: str
    result: str

class Routes(BaseModel):
    step: Literal["factual", "opinion", "math"] = Field(
        description="Classify the input as factual, opinion, or math"
    )

router = llm.with_structured_output(Routes)

def llm_factual(state: State):
    result = llm.invoke(state["input"])
    return {"result": f"[FACTUAL] {result.content}"}

def llm_opinion(state: State):
    result = llm.invoke(state["input"])
    return {"result": f"[OPINION] {result.content}"}

def llm_math(state: State):
    result = llm.invoke(state["input"])
    return {"result": f"[MATH] {result.content}"}

def llm_router(state: State):
    choice = router.invoke([
        SystemMessage(content="Classify the user query as factual, opinion, or math"),
        HumanMessage(content=state["input"])
    ])
    return {"choice": choice.step}

def final_decision(state: State):
    if state["choice"] == "factual":
        return "llm_factual"
    if state["choice"] == "opinion":
        return "llm_opinion"
    if state["choice"] == "math":
        return "llm_math"

graph = StateGraph(State)
graph.add_node("Factual", llm_factual)
graph.add_node("Opinion", llm_opinion)
graph.add_node("Math", llm_math)
graph.add_node("Router", llm_router)

graph.add_edge(START, "Router")
graph.add_conditional_edges(
    "Router",
    final_decision,
    {"llm_factual": "Factual", "llm_opinion": "Opinion", "llm_math": "Math"}
)
graph.add_edge("Factual", END)
graph.add_edge("Opinion", END)
graph.add_edge("Math", END)

graph_builder = graph.compile()

display(Image(graph_builder.get_graph().draw_mermaid_png()))

response = graph_builder.invoke({"input": "What is the capital of India?"})
print(response["result"])
