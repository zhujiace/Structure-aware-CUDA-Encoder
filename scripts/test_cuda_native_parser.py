import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scem.states import IncrementalTreeSitterCudaAST


DEFAULT_DATASET = PROJECT_ROOT / "external" / "CUDABench" / "Datasets" / "CUDABench-Set.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone validation for full tree-sitter-cuda AST snapshots."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--task-stride", type=int, default=10)
    parser.add_argument("--prefix-fractions", default="0.15,0.35,0.60,1.0")
    parser.add_argument("--output-dir", default="analysis_outputs/cuda_ast/native_ast_260521")
    parser.add_argument("--max-examples", type=int, default=6)
    parser.add_argument("--text-limit", type=int, default=160)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_kernel_source(source: str) -> str:
    start = source.find("__global__")
    if start < 0:
        return source
    brace_start = source.find("{", start)
    if brace_start < 0:
        return source[start:]
    depth = 0
    for index in range(brace_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    return source[start:]


def top_counts(counter: Counter, limit: int = 20) -> Dict[str, int]:
    return {key: count for key, count in counter.most_common(limit)}


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    node_type_counts = Counter()
    edge_type_counts = Counter()
    for row in rows:
        node_type_counts.update(row["node_type_counts"])
        edge_type_counts.update(row["edge_type_counts"])
    return {
        "count": len(rows),
        "root_error_rate": mean(float(row["root_has_error"]) for row in rows),
        "incremental_reuse_rate": mean(float(row["incremental_reused"]) for row in rows),
        "mean_source_bytes": mean(float(row["source_bytes"]) for row in rows),
        "mean_node_count": mean(float(row["node_count"]) for row in rows),
        "mean_edge_count": mean(float(row["edge_count"]) for row in rows),
        "top_node_types": top_counts(node_type_counts),
        "top_edge_types": top_counts(edge_type_counts),
    }


def main():
    args = parse_args()
    fractions = [float(item.strip()) for item in args.prefix_fractions.split(",") if item.strip()]
    records = load_jsonl(Path(args.dataset))[:: max(1, args.task_stride)]
    records = records[: args.limit]

    parser = IncrementalTreeSitterCudaAST(include_text=True, text_limit=args.text_limit)
    rows = []
    examples = []
    by_fraction: Dict[str, List[Dict[str, Any]]] = {str(fraction): [] for fraction in fractions}
    for record in records:
        kernel = extract_kernel_source(str(record.get("bench.cu", "")))
        parser.reset()
        for fraction in fractions:
            prefix_len = max(1, int(len(kernel) * fraction))
            snapshot = parser.parse(kernel[:prefix_len], incremental=True)
            row = {
                "task_id": record.get("id"),
                "task_name": record.get("task_name"),
                "fraction": fraction,
                **snapshot.summary_dict(),
            }
            rows.append(row)
            by_fraction[str(fraction)].append(row)
            if len(examples) < args.max_examples and fraction in (fractions[0], fractions[-1]):
                examples.append(
                    {
                        "task_id": record.get("id"),
                        "task_name": record.get("task_name"),
                        "fraction": fraction,
                        "prefix_preview": " ".join(kernel[: min(prefix_len, 260)].split()),
                        "ast": snapshot.as_dict(),
                    }
                )

    summary = {
        "dataset": str(Path(args.dataset)),
        "records": len(records),
        "task_stride": args.task_stride,
        "fractions": fractions,
        "provider": parser.provider,
        "overall": summarize_rows(rows),
        "by_fraction": {fraction: summarize_rows(fraction_rows) for fraction, fraction_rows in by_fraction.items()},
        "examples": examples,
    }
    print(json.dumps(summary, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'rows.jsonl'}")


if __name__ == "__main__":
    main()
