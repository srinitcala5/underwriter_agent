# Underwriter Agent

Multi-turn agentic AI underwriting assistant built with LangGraph.

## Setup

1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.envexample` to `.env` and add your OpenAI API key

## Run

python run.py --record-id 168460-43865 --pdf InputSample.pdf

## Deliverables

- `transcript_168460-43865.txt` — five turn conversation transcript
- `summary.json` — structured underwriting summary
- `ARCHITECTURE.md` — design decisions and production recommendations
