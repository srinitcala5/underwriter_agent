# Architecture Notes — Underwriter Agent
## Assessment: Multi-Turn Agentic AI Underwriting Assistant

---

## Overview

A multi-turn agentic AI assistant built with LangGraph that allows an underwriter
to converse with a life insurance application PDF. The agent maintains context
across turns, uses tools to search and analyse the document, and validates every
response through a parallel judge subgraph before returning it to the user.

Run command:
python run.py --record-id 168460-43865 --pdf InputSample.pdf

---

## 1. PDF Extraction

Library chosen: pdfplumber

Approach:
Text and tables are extracted separately per page and sorted by vertical
y-coordinate to preserve natural reading order. Without sorting, table content
floats out of context and breaks RAG retrieval quality. For example if a table
about occupation details appears before the text that introduces it, the FAISS
chunks lose the surrounding context that gives the table meaning.

Tables are extracted as pipe-separated rows using find_tables() which returns
bounding box coordinates. Plain text is extracted from non-table regions by
filtering out objects that fall within table bounding boxes, preventing
duplication. Both are given their vertical y-coordinate and sorted together
before chunking.

Known limitations:

Checkbox state: The input PDF uses checkbox fields extensively for medical
history, recreational activities, and disclosures. pdfplumber extracts checkbox
label text but cannot determine checked or unchecked state. This causes false
positives. For example the label Breast cancer extracts as text even though
that checkbox is unticked. The agent may incorrectly flag unchecked conditions
as applicant disclosures. This was observed during testing where the agent
incorrectly flagged breast cancer as a red flag.

Production fix: Azure Document Intelligence (Form Recognizer) detects checkbox
state natively and returns selected or unselected markers per field, eliminating
this class of error entirely.

FontBBox warnings: The source PDF contains malformed font descriptors which
cause pdfplumber to emit Could not get FontBBox warnings during extraction.
These are cosmetic and do not affect text extraction quality.

Full extraction warnings logged to extraction_warnings.log at runtime.
19 tables detected across 11 pages, all extracted as pipe-separated rows
with visual structure noted as not preserved.

Handwriting: Handwritten or scanned content would not extract at all with
pdfplumber. Azure Document Intelligence handles this natively.

---

## 2. Chunking and Embedding

Chunking: RecursiveCharacterTextSplitter, 500 characters per chunk, 50
character overlap. Overlap ensures context is not lost at chunk boundaries.
If a sentence spans two chunks the overlap means both chunks contain enough
surrounding text for the embedding to represent the meaning correctly.

Chunk size is configurable via environment variables CHUNK_SIZE and
CHUNK_OVERLAP in .env so it can be tuned without code changes.

Embedding model: OpenAI text-embedding-3-small. Fast and cost-effective for
structured insurance application text where terminology is consistent.

Vector store: FAISS in-memory index built once at startup, queried per turn.
The index does not persist between process restarts. Every run rebuilds from
the PDF.

Production recommendation:
Replace FAISS with Azure AI Search for persistence across restarts, horizontal
scalability, metadata filtering by record_id, and native integration with Azure
OpenAI. Use text-embedding-3-large for higher dimensional embeddings and better
semantic accuracy on medical and legal terminology. Pre-compute and cache
embeddings per application record to avoid rebuild cost on every process start.

---

## 3. Agent State Design

AgentState is a TypedDict that flows through every node in the graph:

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        record_id: str
        draft_response: str | None
        source_chunks: Annotated[list[str], operator.add]
        judge_verdict: dict | None
        revision_count: int

Explicit reducers are used on two fields:

messages uses add_messages which appends new messages rather than overwriting
the list. This preserves the full conversation history across turns.

source_chunks uses operator.add which appends chunks from each tool call rather
than overwriting. When the agent calls multiple tools in one turn, all retrieved
chunks accumulate in state for the judges to validate against.

Without explicit reducers, parallel node writes overwrite each other and data
is lost. This is particularly important for the judge subgraph where three nodes
write their results simultaneously.

draft_response holds the candidate answer between the agent node and judge
subgraph. It is only populated when the LLM returns a final reply with no tool
calls pending.

revision_count is bounded to prevent infinite revision loops. After one failed
revision attempt the response is emitted with a visible warning block.

---

## 4. Three Tools

document_search(query: str) -> str
Queries the FAISS index and returns the top four most relevant chunks. Chunks
are formatted with markers so they can be extracted from ToolMessage content
and passed to the judge subgraph for validation.

underwriting_analysis(topic: str) -> dict
Retrieves chunks for a specific topic then calls the LLM with an underwriting-
focused system prompt applying rules for BMI thresholds, hazardous activity
loadings, cancer staging requirements, alcohol counselling history, and date
inconsistencies. Returns structured JSON with findings, red flags, risk level
and follow_up_required.

summary_generator() -> dict
Queries all major sections of the application across thirteen targeted queries,
builds full context, and calls the LLM to produce a structured JSON summary
matching the Output Sample section layout. Automatically saves to summary.json
on disk when called.

set_retriever pattern:
Tools are imported before the FAISS index is built. A module-level variable
with a setter function solves this dependency ordering problem. run.py builds
the retriever first, then calls set_retriever to inject it into the tools
module. All three tools share the same FAISS index. Equivalent to constructor
injection in C#.

temperature=0 on all LLM calls for deterministic output. Underwriting decisions
need to be consistent and auditable. The same input always produces the same
output.

---

## 5. Judge Subgraph — Parallel Fan-Out

Three judges run concurrently in the same LangGraph superstep:

    fan_out
    |-- factuality_judge   checks claims exist in source chunks
    |-- completeness_judge checks all relevant disclosures are covered
    |-- consistency_judge  checks for contradictions and date conflicts
            |
      merge_verdicts       combines all three into final verdict

Why parallel and not sequential:
Sequential judges add latency proportional to the number of judges. Three
judges each taking one second means three seconds sequentially. Parallel
execution means one second regardless of how many judges are added. The spec
specifically called out LangGraph native fan-out rather than a sequential
for-loop as one of the core things being evaluated.

How parallelism works:
When edges are added from one node to multiple sibling nodes, LangGraph puts
all siblings in the same superstep and executes them concurrently. The fan_out
node is a pass-through that triggers the three judges simultaneously.

The operator.add reducer on judge_results merges partial writes from all three
judges before merge_verdicts runs. Each judge appends its result to the list.
By the time merge_verdicts executes, all three results are present regardless
of execution order.

Verdict structure:
    {
        "overall_pass": false,
        "factuality": {"pass": true, "issues": []},
        "completeness": {"pass": false, "issues": ["Missing alcohol history"]},
        "consistency": {"pass": true, "issues": []},
        "confidence_score": 0.85
    }

Confidence score drops 0.15 per failed judge. One failure gives 0.85, two
gives 0.70, all three gives 0.55.

Production note: Add more judge nodes with edges from fan_out to scale to N
judges without changing the merge logic.

---

## 6. Source Chunks for Judge Validation

After the agent produces a final answer, source_chunks in AgentState is
populated by extracting content from ToolMessage objects in the conversation
history. document_search returns chunks with markers which are parsed and
collected. underwriting_analysis and summary_generator return JSON which is
treated as a single chunk.

The judges receive these chunks as ground truth for validation. Without
populated source_chunks, judges default to pass with a note explaining no
chunks were available, making validation meaningless.

---

## 7. Revise-or-Flag Control Flow

    agent -> (tool calls?) -> tools -> agent
    agent -> (final reply?) -> judges -> merge
    merge -> (overall_pass?) -> emit
    merge -> (failed, revision_count < 1?) -> agent with judge feedback
    merge -> (failed, revision_count >= 1?) -> emit with warning block

When routing back to the agent for revision, the judge verdict issues are
appended as a system message so the agent knows specifically what to fix
rather than revising blindly.

Revision is bounded at one attempt. If the revised response still fails judges,
it emits with a visible warning block prefixed with the judge flag symbol.
emit_node resets revision_count to zero for the next turn.

---

## 8. Memory and Checkpointing

Assessment: MemorySaver, in-memory checkpointer. Conversation history is lost
on process restart. Each application gets its own conversation thread keyed on
record_id so the agent remembers context across turns within a session.

Production recommendation: SqliteSaver for single-instance deployments.
PostgresSaver for multi-instance or cloud deployments where multiple pods
handle requests concurrently and need shared conversation state.

---

## 9. What I Would Change for Production

Replace FAISS with Azure AI Search for persistent scalable vector storage
with metadata filtering.

Replace MemorySaver with PostgresSaver for conversation persistence across
restarts and horizontal scaling.

Replace pdfplumber with Azure Document Intelligence for checkbox state
detection, table extraction with structural fidelity, and handwriting support.

Add structured JSON logging of every node transition for full auditability.
Insurance underwriting decisions need complete audit trails.

Add exponential backoff retry on all LLM calls using tenacity.

Add human-in-the-loop interrupt before emitting flagged responses so the
underwriter can accept or reject the agent recommendation.

Expose the graph as a FastAPI endpoint with streaming via astream_events.

Pre-compute and cache FAISS embeddings per application record.

Add LangSmith tracing for full observability across agent turns including
which tools were called, LLM inputs and outputs, and judge verdicts.