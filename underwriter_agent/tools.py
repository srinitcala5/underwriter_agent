import os
import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# retriever will be injected at runtime from graph.py
# we use a module-level variable so tools can access it
retriever = None


def set_retriever(r):
    """Called once at startup from run.py after FAISS index is built."""
    global retriever
    retriever = r


@tool
def document_search(query: str) -> str:
    """
    Search the loaded insurance application PDF for passages relevant to the query.
    Always call this first before analysing any topic.
    Returns relevant text chunks from the document.
    """
    if retriever is None:
        return "Error: retriever not initialised. Call set_retriever() first."

    # search FAISS index for relevant chunks
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant content found in the document for this query."

    # join chunks with separator so agent sees all relevant passages
    chunks = []
    for i, doc in enumerate(docs):
        chunks.append(f"[Chunk {i+1}]\n{doc.page_content}")

    return "\n\n".join(chunks)


@tool
def underwriting_analysis(topic: str) -> dict:
    """
    Analyse a specific underwriting topic from the insurance application.
    Topic examples: 'medical history', 'travel disclosures', 'hazardous activities',
    'smoking', 'alcohol', 'occupation', 'BMI'.
    Returns findings, red flags, risk level and whether follow-up is required.
    """
    if retriever is None:
        return {"error": "retriever not initialised"}

    # step 1: retrieve relevant chunks for this topic
    docs = retriever.invoke(topic)
    if not docs:
        return {
            "findings": "No relevant content found",
            "red_flags": [],
            "risk_level": "unknown",
            "follow_up_required": False
        }

    # step 2: build context from retrieved chunks
    context = "\n\n".join(doc.page_content for doc in docs)

    # step 3: call LLM with underwriting analyst system prompt
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    system_prompt = (
        "You are an experienced life insurance underwriter.\n"
        "Analyse the provided application excerpts for the given topic.\n"
        "Apply these underwriting rules:\n"
        "- BMI over 30 is elevated risk\n"
        "- Smoker status increases premium\n"
        "- Hazardous activities (scuba diving, aviation, racing) require loadings\n"
        "- Cancer history requires staging and recurrence assessment\n"
        "- Alcohol counselling history is a red flag\n"
        "- Unresolved medical investigations are red flags\n"
        "- Date inconsistencies in travel or leave declarations are red flags\n\n"
        "Return a JSON object with exactly these fields:\n"
        "{\n"
        '    "findings": "summary of what you found",\n'
        '    "red_flags": ["list", "of", "red", "flags"],\n'
        '    "risk_level": "low or medium or high or decline",\n'
        '    "follow_up_required": true or false\n'
        "}\n"
        "Return only valid JSON. No explanation outside the JSON."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Topic: {topic}\n\nApplication excerpts:\n{context}")
    ]

    response = llm.invoke(messages)

    # step 4: parse JSON response
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "findings": response.content,
            "red_flags": [],
            "risk_level": "unknown",
            "follow_up_required": True
        }

    return result


@tool
def summary_generator() -> dict:
    """
    Generate the full structured underwriting summary for the loaded application.
    Call this only when the user explicitly requests the final summary.
    Returns a complete assessment mirroring the Output Sample section layout.
    """
    if retriever is None:
        return {"error": "retriever not initialised"}

    # retrieve content for each major section
    sections_to_retrieve = [
        "applicant personal details date of birth occupation",
        "sum insured life TPD trauma income protection cover",
        "existing insurance other insurer replacing",
        "residency visa Australia",
        "income employment annual salary",
        "travel overseas destination duration dates",
        "recreation hazardous activities scuba diving aviation cycling",
        "alcohol drinks counselling",
        "smoking cigarettes",
        "BMI height weight",
        "medical history cancer melanoma heart cholesterol migraines",
        "family history cancer disease",
        "GP doctor details practice"
    ]

    # build full context from all section queries
    full_context = ""
    for query in sections_to_retrieve:
        docs = retriever.invoke(query)
        for doc in docs:
            full_context += doc.page_content + "\n"

    # call LLM to generate structured summary
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    system_prompt = (
        "You are an experienced life insurance underwriter.\n"
        "Generate a complete structured underwriting summary from the application excerpts.\n"
        "Return a JSON object with exactly these sections:\n"
        "{\n"
        '    "applicant": {"name": "", "dob": "", "age": 0, "gender": "", "state": "", "occupation": "", "occupation_class": "", "smoker": true, "bmi": 0.0},\n'
        '    "applied_for_cover": {"life": 0, "tpd": 0, "trauma": 0, "income_protection": 0},\n'
        '    "existing_cover": {"description": ""},\n'
        '    "modified_terms": null,\n'
        '    "claims_history": "",\n'
        '    "residency": "",\n'
        '    "occupation": {"primary": "", "secondary": ""},\n'
        '    "income": {"primary": 0, "secondary": 0, "total": 0},\n'
        '    "travel": {"planned": true, "destination": "", "duration": "", "departure": "", "leave_start": ""},\n'
        '    "recreation": [],\n'
        '    "alcohol": {"drinks_per_week": 0, "counselling_history": true, "hospitalised": false},\n'
        '    "drug_use": "",\n'
        '    "smoking": {"smoker": true, "cigarettes_per_day": ""},\n'
        '    "bmi": {"height_cm": 0, "weight_kg": 0, "bmi": 0.0, "category": ""},\n'
        '    "medical_history": {},\n'
        '    "family_history": {},\n'
        '    "gp_details": {"practice": "", "address": "", "patient_duration": ""},\n'
        '    "red_flags": [],\n'
        '    "disclosure_cross_check": []\n'
        "}\n"
        "Return only valid JSON. No explanation outside the JSON."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Application excerpts:\n{full_context}")
    ]

    response = llm.invoke(messages)

    # parse and return
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {"error": "Failed to parse summary", "raw": response.content}

    # parse and return
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {"error": "Failed to parse summary", "raw": response.content}

    # save summary.json to disk automatically
    try:
        output_path = os.path.join(os.getcwd(), "summary.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("[summary_generator] summary.json saved to disk")
    except Exception as e:
        print(f"[summary_generator] Could not save summary.json: {e}")

    return result