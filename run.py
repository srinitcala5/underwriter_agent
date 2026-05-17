import argparse
import json
import os
from dotenv import load_dotenv

from underwriter_agent.retriever import build_retriever
from underwriter_agent.tools import set_retriever
from underwriter_agent.graph import build_graph

load_dotenv()


def main():
    # step 1: read command line arguments
    parser = argparse.ArgumentParser(description="Underwriter Agent CLI")
    parser.add_argument("--record-id", required=True, help="Application record ID")
    parser.add_argument("--pdf", required=True, help="Path to insurance application PDF")
    args = parser.parse_args()

    print(f"\n[Underwriter Agent] Loading application {args.record_id}...")

    # step 2: build FAISS retriever from PDF
    retriever = build_retriever(args.pdf)

    # step 3: inject retriever into tools
    set_retriever(retriever)

    # step 4: build the graph
    graph = build_graph()

    # step 5: configure thread for this record
    # thread_id keyed on record_id so each application
    # has its own conversation memory
    config = {"configurable": {"thread_id": args.record_id}}

    print(f"[Underwriter Agent] Ready. Application {args.record_id} loaded.")
    print("[Underwriter Agent] Type 'exit' to quit.\n")

    # step 6: open transcript file once before the loop
    # transcript captures full conversation for submission
    transcript_path = f"transcript_{args.record_id}.txt"

    with open(transcript_path, "w", encoding="utf-8") as transcript:
        transcript.write(f"=== Underwriter Agent Session ===\n")
        transcript.write(f"Record ID: {args.record_id}\n")
        transcript.write(f"PDF: {args.pdf}\n")
        transcript.write(f"================================\n\n")

        # step 7: start chat loop
        while True:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("[Underwriter Agent] Goodbye.")
                transcript.write("\n[Session ended]\n")
                break

            # write user question to transcript
            transcript.write(f"> {user_input}\n\n")

            # step 8: send message to graph
            result = graph.invoke(
                {
                    "messages": [("user", user_input)],
                    "record_id": args.record_id,
                    "draft_response": None,
                    "source_chunks": [],
                    "judge_verdict": None,
                    "revision_count": 0
                },
                config=config
            )

            # step 9: extract and print the last assistant message
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                print(f"\n[Agent] {last_message.content}\n")

                # show judge verdict if available
                verdict = result.get("judge_verdict")
                if verdict:
                    confidence = verdict.get("confidence_score", 1.0)
                    overall = "PASS" if verdict.get("overall_pass") else "FAIL"
                    print(f"[Judge] {overall} | Confidence: {confidence}\n")

                # write agent response to transcript
                transcript.write(f"[Agent] {last_message.content}\n\n")

                if verdict:
                    transcript.write(
                        f"[Judge] {overall} | Confidence: {confidence}\n\n"
                    )

                transcript.write("---\n\n")

                

            # flush transcript after every turn
            # ensures nothing is lost if program crashes
            transcript.flush()

    print(f"[Underwriter Agent] Transcript saved to {transcript_path}")


if __name__ == "__main__":
    main()