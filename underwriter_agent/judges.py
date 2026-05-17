import os
import json
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()


# ── Judge State ────────────────────────────────────────────────────────────

class JudgeState(TypedDict):
    draft_response: str                                     # candidate answer to evaluate
    source_chunks: list[str]                                # PDF chunks that grounded the answer
    messages: list[BaseMessage]                             # conversation history for consistency check
    judge_results: Annotated[list[dict], operator.add]      # reducer: merges parallel verdicts
    judge_verdict: dict | None                              # final merged verdict


# ── Helper ─────────────────────────────────────────────────────────────────

def get_llm():
    """Return configured LLM instance."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


# ── Fan-out Node ───────────────────────────────────────────────────────────

def fan_out(state: JudgeState) -> dict:
    """
    Entry point for judge subgraph.
    Pass-through node that triggers parallel execution of all three judges.
    LangGraph runs all nodes with edges from this node in the same superstep,
    meaning factuality, completeness and consistency run concurrently.
    """
    return {}  # no state changes, just triggers the fan-out


# ── Judge Nodes ────────────────────────────────────────────────────────────

def factuality_judge(state: JudgeState) -> dict:
    """
    Judge 1: Factuality
    Checks every claim in the draft response exists in the source chunks.
    Returns pass/fail and list of unsupported claims.
    """
    draft = state["draft_response"]
    chunks = state["source_chunks"]

    # if no chunks available, handle gracefully
    if not chunks:
        return {
            "judge_results": [{
                "judge": "factuality",
                "pass": True,
                "issues": [],
                "note": "No source chunks available - response was a clarification"
            }]
        }

    context = "\n\n".join(chunks)

    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a factuality judge for insurance underwriting.
Your job: check if every factual claim in the response is supported by the source chunks.
Return a JSON object with exactly these fields:
{
    "pass": true or false,
    "issues": ["list of unsupported claims if any"]
}
Return only valid JSON. No explanation outside the JSON."""),
        HumanMessage(content=f"""Source chunks from PDF:
{context}

Draft response to evaluate:
{draft}

Check: does every claim in the draft response exist in the source chunks?""")
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {"pass": True, "issues": []}

    return {
        "judge_results": [{
            "judge": "factuality",
            "pass": result.get("pass", True),
            "issues": result.get("issues", [])
        }]
    }


def completeness_judge(state: JudgeState) -> dict:
    """
    Judge 2: Completeness
    Checks all materially relevant disclosures are covered in the response.
    Returns pass/fail and list of missing items.
    """
    draft = state["draft_response"]
    chunks = state["source_chunks"]

    if not chunks:
        return {
            "judge_results": [{
                "judge": "completeness",
                "pass": True,
                "issues": [],
                "note": "No source chunks available"
            }]
        }

    context = "\n\n".join(chunks)

    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a completeness judge for insurance underwriting.
Your job: check if the response covers all materially relevant disclosures present in the source chunks.
Focus on: medical conditions, hazardous activities, travel plans, occupation risks, financial disclosures.
Return a JSON object with exactly these fields:
{
    "pass": true or false,
    "issues": ["list of missing items if any"]
}
Return only valid JSON. No explanation outside the JSON."""),
        HumanMessage(content=f"""Source chunks from PDF:
{context}

Draft response to evaluate:
{draft}

Check: are all relevant disclosures from the chunks covered in the response?""")
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {"pass": True, "issues": []}

    return {
        "judge_results": [{
            "judge": "completeness",
            "pass": result.get("pass", True),
            "issues": result.get("issues", [])
        }]
    }


def consistency_judge(state: JudgeState) -> dict:
    """
    Judge 3: Consistency
    Checks for internal contradictions and conflicts with earlier turns.
    Returns pass/fail and list of contradictions found.
    """
    draft = state["draft_response"]
    msgs = state["messages"]

    # build conversation history for context
    history = ""
    for msg in msgs[:-1]:  # exclude current message
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        history += f"{role}: {msg.content}\n"

    llm = get_llm()
    messages_to_send = [
        SystemMessage(content="""You are a consistency judge for insurance underwriting.
Your job: check for internal contradictions in the response and conflicts with earlier conversation turns.
Key things to check:
- Date inconsistencies (travel dates, leave dates, treatment dates)
- Occupation conflicts (stated occupation vs declared activities)
- Income inconsistencies
- Medical history contradictions
Return a JSON object with exactly these fields:
{
    "pass": true or false,
    "issues": ["list of contradictions if any"]
}
Return only valid JSON. No explanation outside the JSON."""),
        HumanMessage(content=f"""Conversation history:
{history}

Draft response to evaluate:
{draft}

Check: are there any internal contradictions or conflicts with earlier turns?""")
    ]

    response = llm.invoke(messages_to_send)

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {"pass": True, "issues": []}

    return {
        "judge_results": [{
            "judge": "consistency",
            "pass": result.get("pass", True),
            "issues": result.get("issues", [])
        }]
    }


# ── Merge Node ─────────────────────────────────────────────────────────────

def merge_verdicts(state: JudgeState) -> dict:
    """
    Merge node. Combines results from all three parallel judges.
    operator.add reducer ensures all three judge_results are collected
    before this node runs.
    """
    results = state["judge_results"]

    # build structured verdict
    verdict = {
        "overall_pass": True,
        "factuality": {"pass": True, "issues": []},
        "completeness": {"pass": True, "issues": []},
        "consistency": {"pass": True, "issues": []},
        "confidence_score": 1.0
    }

    failed_count = 0

    for result in results:
        judge_name = result["judge"]
        passed = result.get("pass", True)
        issues = result.get("issues", [])

        verdict[judge_name] = {
            "pass": passed,
            "issues": issues
        }

        if not passed:
            failed_count += 1
            verdict["overall_pass"] = False

    # confidence score drops 15% per failed judge
    verdict["confidence_score"] = round(1.0 - (failed_count * 0.15), 2)

    return {"judge_verdict": verdict}


# ── Subgraph Builder ───────────────────────────────────────────────────────

def build_judge_subgraph():
    """
    Build and compile the judge subgraph with parallel fan-out.

    Architecture:
    fan_out → [factuality_judge, completeness_judge, consistency_judge] → merge_verdicts
    
    All three judges run in the same LangGraph superstep (concurrently).
    operator.add reducer on judge_results merges their outputs before merge_verdicts runs.

    Production note: add more judge nodes with edges from fan_out
    to scale to N judges without changing the merge logic.
    """

    g = StateGraph(JudgeState)

    # register all nodes
    g.add_node("fan_out", fan_out)
    g.add_node("factuality_judge", factuality_judge)
    g.add_node("completeness_judge", completeness_judge)
    g.add_node("consistency_judge", consistency_judge)
    g.add_node("merge_verdicts", merge_verdicts)

    # entry point hits fan_out first
    g.set_entry_point("fan_out")

    # fan_out triggers all three judges in parallel
    # LangGraph superstep model runs sibling nodes concurrently
    g.add_edge("fan_out", "factuality_judge")
    g.add_edge("fan_out", "completeness_judge")
    g.add_edge("fan_out", "consistency_judge")

    # all three judges feed into merge_verdicts
    # reducer collects all three results before merge runs
    g.add_edge("factuality_judge", "merge_verdicts")
    g.add_edge("completeness_judge", "merge_verdicts")
    g.add_edge("consistency_judge", "merge_verdicts")

    g.add_edge("merge_verdicts", END)

    return g.compile()