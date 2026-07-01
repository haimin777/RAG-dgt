import argparse
import json
import time
from pathlib import Path

from rag import answer_query


def build_query(data: dict) -> str:
    question = (data.get("question") or "").strip()
    options = data.get("options") or []

    lines = [
        "Analyze the driver's theory test question (Spanish Permiso B / DGT).",
        "Select the correct option and briefly explain why.",
        "",
        f"Question: {question}",
    ]
    if options:
        lines.append("")
        lines.extend([opt for opt in options if isinstance(opt, str)])
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-only test runner")
    parser.add_argument("json_file", help="Path to parsed JSON file")
    args = parser.parse_args()

    data = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    query = build_query(data)

    t_rag_start = time.perf_counter()
    response = answer_query(query)
    t_rag = time.perf_counter() - t_rag_start

    print(response)
    print(f"\nTiming: rag={t_rag:.2f}s")


if __name__ == "__main__":
    main()
