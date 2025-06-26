from typing import List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain.schema import HumanMessage
import base64
from nodes import (
    skills_extraction_node,
    skill_check_node,
    skill_grader_node,
    hire_decision_node,
    reject_cv_node,
)


# --- Graph State ---
class GraphState(TypedDict):
    model: object  # You can refine this to `BaseLanguageModel` if you're using LangChain LLMs
    base64_image: base64
    extracted_skills: Optional[dict]
    scored_skills: Optional[dict]
    is_skill_met: Optional[bool]
    is_hired: Optional[bool]
    response: Optional[HumanMessage]


# --- Fallback Node ---
def handle_unrelated_content(state: GraphState) -> GraphState:
    state["response"] = HumanMessage(
        content="The query appears to be out of scope for this system."
    )
    return state


# --- Build Graph ---
def build_graph():
    workflow = StateGraph(GraphState)

    # Nodes (must be defined elsewhere in your code)
    workflow.add_node("skills_extraction", skills_extraction_node)
    workflow.add_node("skill_check", skill_check_node)
    workflow.add_node("skill_grader", skill_grader_node)
    workflow.add_node("hire_decision", hire_decision_node)
    workflow.add_node("rejected", reject_cv_node)
    workflow.add_node("out_of_scope", handle_unrelated_content)

    # Edges
    workflow.add_edge(START, "skills_extraction")
    workflow.add_edge("skills_extraction", "skill_check")

    workflow.add_conditional_edges(
        "skill_check",
        lambda state: "yes" if state["is_skill_met"] else "no",
        {"yes": "skill_grader", "no": "rejected"},
    )

    workflow.add_edge("skill_grader", "hire_decision")

    workflow.add_conditional_edges(
        "hire_decision",
        lambda state: "yes" if state.get("is_hired") else "no",
        {"yes": END, "no": "rejected"},
    )

    workflow.add_edge("rejected", END)
    workflow.add_edge("out_of_scope", END)

    return workflow.compile()
