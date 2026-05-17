import os
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from underwriter_agent.judges import build_judge_subgraph

load_dotenv()

from underwriter_agent.tools import (
    document_search,
    underwriting_analysis,
    summary_generator,
    set_retriever
)


# ── State ──────────────────────────────────────────────────────────────────
# TypedDict tells LangGraph exactly what fields live in state
# and how to merge updates from parallel nodes (via reducers)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # reducer: appends messages
    record_id: str                                         # which application we're on
    draft_response: str | None                             # candidate answer pre-judge
    source_chunks: Annotated[list[str], operator.add]      # reducer: appends chunks
    judge_verdict: dict | None                             # merged verdict from judges
    revision_count: int                                    # bounded to avoid loops


# ── Agent Node ─────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """
    Core agent node.
    1. Call LLM bound to 3 tools
    2. If LLM returns tool calls -> return tool call message, ToolNode handles execution
    3. If LLM returns final reply -> write to draft_response for judge subgraph
    """

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # bind tools to LLM so it knows what it can call
    tools = [document_search, underwriting_analysis, summary_generator]
    llm_with_tools = llm.bind_tools(tools)

    # system prompt tells agent how to behave
    system_message = SystemMessage(content="""You are an expert life insurance underwriting assistant.
You have access to a loaded insurance application PDF.

When asked about date inconsistencies:
- Search for "extended leave date reason duration"
- Search for "overseas travel departure date country"
- Search for "planned leave work absence"
- Compare all dates found and flag any conflicts

When asked about red flags or medical history:
- Search multiple times with different queries
- Search for each risk category separately

Always use document_search to find relevant passages before answering.
Use underwriting_analysis to assess specific risk topics.
Use summary_generator only when user explicitly asks for final summary.
Never make up facts not present in the application.
If information is not in the document, say so clearly.""")

    # build message list with system prompt prepended
    messages = [system_message] + state["messages"]

    # call LLM
    response = llm_with_tools.invoke(messages)

    # if LLM made tool calls, return as-is for ToolNode to handle
    if response.tool_calls:
        return {"messages": [response]}
    
    # extract source chunks from tool messages in conversation history
    # so judges can validate the response against actual PDF content
    source_chunks = []
        
    for message in state["messages"]:
        # tool messages contain the raw tool output
        #if hasattr(message, "type") and message.type == "ToolMessage":
        if isinstance(message,ToolMessage):
            content = message.content
            if content and isinstance(content, str):
                # each chunk is separated by our chunk separator
                if "[Chunk" in content:
                    # split on chunk markers and collect
                    parts = content.split("[Chunk")
                    for part in parts[1:]:  # skip first empty split
                        chunk_text = part.split("]", 1)[-1].strip()
                        if chunk_text:
                            source_chunks.append(chunk_text)
                else:
                    source_chunks.append(content)

      
    # if LLM returned final answer, write to draft_response for judges
    return {
    "messages": [response],
    "draft_response": response.content,
    "source_chunks": source_chunks
}


# ── Routing Functions ───────────────────────────────────────────────────────

def route_after_agent(state: AgentState) -> str:
    """
    After agent node runs, decide where to go next.
    If last message has tool calls -> go to tools node
    If last message is final reply -> go to judges
    """
    last_message = state["messages"][-1]

    # check if LLM wants to call a tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # no tool calls means final answer ready for judging
    return "judges"


def route_after_judges(state: AgentState) -> str:
    """
    After judges run, decide where to go next.
    If verdict passes -> emit response to user
    If verdict fails and revision count < 1 -> loop back to agent to revise
    If verdict fails and revision count >= 1 -> emit anyway to avoid infinite loop
    """
    verdict = state.get("judge_verdict")
    revision_count = state.get("revision_count", 0)

    # guard against None verdict
    if verdict is None:
        return "emit"

    if verdict.get("overall_pass", True):
        return "emit"

    if revision_count < 1:
        return "agent"  # one revision attempt

    return "emit"  # emit with warning after one attempt


# ── Emit Node ──────────────────────────────────────────────────────────────

def emit_node(state: AgentState) -> dict:
    """
    Final node. Emits response to user.
    If judges flagged issues and we exhausted revisions, add warning block.
    """
    verdict = state.get("judge_verdict")
    draft = state.get("draft_response", "")

    # guard against None verdict
    if verdict is None:
        return {
            "messages": [AIMessage(content=draft)],
            "revision_count": 0
        }

    if not verdict.get("overall_pass", True):
        # collect all issues from failed judges
        issues = []
        for judge in ["factuality", "completeness", "consistency"]:
            judge_result = verdict.get(judge, {})
            if not judge_result.get("pass", True):
                issues.extend(judge_result.get("issues", []))

        warning = "\n\n■ Judge flagged: " + "; ".join(issues)
        final_response = draft + warning
    else:
        final_response = draft

    return {
        "messages": [AIMessage(content=final_response)],
        "revision_count": 0  # reset for next turn
    }

# ---- Judege Node --------------------------------------------------------
# ── judges wrapper node ────────────────────────────────────────────────
    # the judge subgraph has its own internal state (JudgeState)
    # this wrapper bridges parent AgentState to JudgeState and back
def judges_node(state: AgentState) -> dict:
    """
    Runs the judge subgraph and writes verdict back to parent state.
    Extracts what judges need from AgentState, passes to subgraph,
    returns judge_verdict for AgentState to store.
    """
    subgraph = build_judge_subgraph()

    # pass only what judges need from parent state
    result = subgraph.invoke({
      "draft_response": state.get("draft_response", ""),
            "source_chunks": state.get("source_chunks", []),
            "messages": state.get("messages", []),
            "judge_results": [],
            "judge_verdict": None  # initialise so merge_verdicts can write to it
        })

    

    return {"judge_verdict": result.get("judge_verdict"),
            "revision_count": state.get("revision_count",0)+1   # increment to bound loop
            }

# ── Graph Builder ──────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Compile the LangGraph state machine.
    Nodes: agent, tools, judges, emit
    Edges define the flow between nodes based on routing functions.

    Production note: swap MemorySaver for SqliteSaver or PostgresSaver
    so conversation history persists across process restarts.
    """
    g = StateGraph(AgentState)

    # register nodes
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode([document_search, underwriting_analysis, summary_generator]))
    g.add_node("judges", judges_node)
    g.add_node("emit", emit_node)

    # entry point
    g.set_entry_point("agent")

    # edges
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "judges": "judges"}
    )

    g.add_edge("tools", "agent")  # after tool runs, go back to agent

    g.add_conditional_edges(
        "judges",
        route_after_judges,
        {"agent": "agent", "emit": "emit"}
    )

    g.add_edge("emit", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())