import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    DEFAULT_CUDABENCH_ROOT,
    LocalGenerator,
    compile_code,
    evaluate_functionality,
    extract_code,
    iter_code_versions,
    load_cudabench_helpers,
    load_cudabench_tasks,
    load_done_ids,
    load_generated_results,
    resolve_cudabench_paths,
    safe_div,
    suppress_output,
)


HARNESS_SYSTEM_PROMPT = """
You are generating CUDA kernel code for a fixed CUDABench harness.

Output exactly one fenced cpp code block and nothing else. The single code block
must contain the replacement `__global__` kernel with the requested kernel name
and compatible parameters. If helper functions are needed, put all helpers before
the kernel in the same code block.

Do not output explanations, analysis, placeholders, `main`, host I/O, includes,
memory allocation, cudaMemcpy calls, kernel launch code, or multiple code blocks.
""".strip()


@dataclass(frozen=True)
class HarnessParts:
    prefix: str
    reference_kernel: str
    suffix: str
    kernel_signature: str
    main_function: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate kernel-only generation by inserting generated kernels into CUDABench bench.cu harnesses."
    )
    parser.add_argument("--cudabench-root", default=str(DEFAULT_CUDABENCH_ROOT))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model-path", default="./models/Qwen3.5-0.8B")
    parser.add_argument("--scem-checkpoint", default=None, help="Optional path to scem.pt for decoding-time SCEM bias.")
    parser.add_argument("--lora-checkpoint", default=None, help="Optional path to a PEFT/LoRA adapter directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory. If omitted, create a unique dated eval_outputs/ harness directory.")
    parser.add_argument("--run-name", default=None, help="Optional label included in the auto-generated output directory name.")
    parser.add_argument("--level", choices=["level1_prompt", "level2_prompt", "level3_prompt"], default="level1_prompt")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--task-stride",
        type=int,
        default=1,
        help="Evaluate every Nth task after --start-index. Use 5 to sample one kernel from each five-task group.",
    )
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--compile-timeout", type=int, default=60)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--trust-generated", action="store_true", help="Skip generation and evaluate an existing JSONL.")
    parser.add_argument("--results-jsonl", default=None, help="Existing generated results JSONL for --trust-generated.")
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "run"


def checkpoint_label(path: Optional[str]) -> str:
    if not path:
        return ""
    p = Path(path)
    if p.name == "scem.pt" and p.parent.name:
        return p.parent.name
    return p.name or p.parent.name


def uniquify_output_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find a free output directory based on {path}")


def build_auto_output_dir(args) -> Path:
    model_label = slugify(Path(args.model_path).name)
    level_label = args.level.replace("_prompt", "")
    mode_parts = ["harness"]
    if args.scem_checkpoint:
        mode_parts.extend(["scem", slugify(checkpoint_label(args.scem_checkpoint))])
    if args.lora_checkpoint:
        mode_parts.extend(["lora", slugify(checkpoint_label(args.lora_checkpoint))])
    if args.task_stride != 1:
        mode_parts.append(f"stride{args.task_stride}")
    if args.limit is not None:
        mode_parts.append(f"limit{args.limit}")
    if args.num_samples != 1:
        mode_parts.append(f"n{args.num_samples}")
    if args.run_name:
        run_label = slugify(args.run_name)
        if run_label not in mode_parts:
            mode_parts.append(run_label)
    date_label = datetime.now().strftime("%y%m%d")
    name = "_".join([level_label, *mode_parts, date_label])
    return uniquify_output_dir(Path("eval_outputs") / model_label / name)


def find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    index = open_index
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError("Unmatched brace in bench.cu")


def find_function_start(text: str, marker_index: int) -> int:
    line_start = text.rfind("\n", 0, marker_index) + 1
    return line_start


def extract_function(text: str, pattern: str) -> Tuple[int, int, str]:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Function pattern not found: {pattern}")
    open_index = text.find("{", match.end() - 1)
    if open_index < 0:
        raise ValueError("Function opening brace not found")
    close_index = find_matching_brace(text, open_index)
    start = find_function_start(text, match.start())
    return start, close_index + 1, text[start : close_index + 1]


def extract_harness_parts(bench_cu: str) -> HarnessParts:
    kernel_match = re.search(r"\b__global__\b", bench_cu)
    if kernel_match is None:
        raise ValueError("bench.cu does not contain a __global__ kernel")
    kernel_start = find_function_start(bench_cu, kernel_match.start())
    kernel_open = bench_cu.find("{", kernel_match.end())
    if kernel_open < 0:
        raise ValueError("Kernel opening brace not found")
    kernel_end = find_matching_brace(bench_cu, kernel_open) + 1
    reference_kernel = bench_cu[kernel_start:kernel_end]
    kernel_signature = bench_cu[kernel_start:kernel_open].strip()

    _, _, main_function = extract_function(bench_cu, r"\bint\s+main\s*\([^)]*\)\s*\{")

    return HarnessParts(
        prefix=bench_cu[:kernel_start],
        reference_kernel=reference_kernel,
        suffix=bench_cu[kernel_end:],
        kernel_signature=kernel_signature,
        main_function=main_function,
    )


def build_harness_source(task: Dict[str, Any], generated_kernel: str) -> str:
    parts = extract_harness_parts(task["bench.cu"])
    return (
        parts.prefix.rstrip()
        + "\n\n// --- BEGIN MODEL GENERATED KERNEL ---\n"
        + generated_kernel.strip()
        + "\n// --- END MODEL GENERATED KERNEL ---\n\n"
        + parts.suffix.lstrip()
    )


def format_io_spec(items: List[Dict[str, Any]]) -> str:
    return "\n".join(f"- {item['name']}: {item['dtype']}, shape = {item['shape']}" for item in items)


def kernel_name_from_signature(signature: str) -> str:
    match = re.search(r"\bvoid\s+([A-Za-z_]\w*)\s*\(", signature)
    if match is None:
        raise ValueError(f"Unable to extract kernel name from signature: {signature}")
    return match.group(1)


def build_harness_prompt(task: Dict[str, Any], level: str, gpu_model: str) -> str:
    parts = extract_harness_parts(task["bench.cu"])
    kernel_name = kernel_name_from_signature(parts.kernel_signature)
    return f"""
Task: write only the CUDA kernel for a fixed test harness.

The evaluator will insert your generated code into the existing harness where
the reference kernel was removed. The harness already contains headers, helper
functions, input/output file handling, memory allocation, cudaMemcpy calls,
kernel launch, and main().

Your output must be exactly one cpp code block containing `{kernel_name}`.
If you need helper functions, put every helper and the kernel in that same code
block. Do not split helpers and kernel across multiple code blocks.

Task Name:
{task['task_name']}

Task Description:
{task[level]}

Input:
{format_io_spec(task['inputs'])}

Output:
{format_io_spec(task['outputs'])}

GPU:
{gpu_model}

Required kernel signature:
{parts.kernel_signature}

Fixed main function supplied by the harness. Do not regenerate it:
BEGIN_FIXED_MAIN
{parts.main_function}
END_FIXED_MAIN

Return only one code block. Optional `__device__` helpers must appear before the
replacement `__global__` kernel in the same code block.
""".strip()


def extract_harness_kernel(response: str, kernel_name: str) -> Tuple[str, List[str]]:
    code = extract_code(
        response,
        required_substrings=["__global__", kernel_name],
        preferred_substrings=[kernel_name, "__global__"],
        forbidden_substrings=["int main", "#include", "cudaMalloc", "cudaMemcpy", "read_binary", "write_binary"],
        fallback_to_best=False,
    )
    errors: List[str] = []
    if not code:
        errors.append("missing_required_kernel_block")
        return "", errors
    if "__global__" not in code:
        errors.append("missing_global")
    if kernel_name not in code:
        errors.append("missing_required_kernel_name")
    if "int main" in code:
        errors.append("contains_main")
    return code, errors


def select_eval_tasks(tasks: List[Dict[str, Any]], stride: int, limit: Optional[int]) -> List[Dict[str, Any]]:
    if stride < 1:
        raise ValueError("--task-stride must be >= 1")
    selected = tasks[::stride]
    if limit is not None:
        selected = selected[:limit]
    return selected


def generate_results(args, tasks: List[Dict[str, Any]], output_path: Path, helpers) -> None:
    generator = LocalGenerator(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        system_prompt=HARNESS_SYSTEM_PROMPT,
        use_scem_prompt=False,
        enable_scem=bool(args.scem_checkpoint),
        scem_checkpoint=args.scem_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        alpha=args.alpha,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path)

    with open(output_path, "a", encoding="utf-8") as handle:
        for index, task in enumerate(tasks, start=1):
            task_id = int(task["id"])
            if task_id in done_ids:
                print(f"[GEN {index}/{len(tasks)}] id={task_id} skip existing")
                continue
            prompt = build_harness_prompt(task, args.level, args.gpu_model)
            parts = extract_harness_parts(task["bench.cu"])
            record = {
                "id": task_id,
                "task_name": task["task_name"],
                "prompt": prompt,
                "level": args.level,
                "model_path": args.model_path,
                "scem_checkpoint": args.scem_checkpoint,
                "lora_checkpoint": args.lora_checkpoint,
                "kernel_signature": parts.kernel_signature,
                "eval_mode": "harness_kernel_only",
            }
            print(f"[GEN {index}/{len(tasks)}] id={task_id} {task['task_name']}")
            kernel_name = kernel_name_from_signature(parts.kernel_signature)
            for sample_idx in range(1, args.num_samples + 1):
                response = generator.generate(
                    prompt,
                    stop_required_substrings=["__global__", kernel_name],
                    stop_forbidden_substrings=[
                        "int main",
                        "#include",
                        "cudaMalloc",
                        "cudaMemcpy",
                        "read_binary",
                        "write_binary",
                    ],
                )
                code, extraction_errors = extract_harness_kernel(response, kernel_name)
                record[f"response{sample_idx}"] = response
                record[f"code{sample_idx}"] = code
                record[f"extraction_errors{sample_idx}"] = extraction_errors
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def evaluate_results(args, tasks_by_id: Dict[int, Dict[str, Any]], results_path: Path, eval_path: Path, helpers) -> Dict[str, Any]:
    records = load_generated_results(results_path)
    temp_root = Path(args.output_dir) / "temp_eval"
    if temp_root.exists() and not args.keep_temp:
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    if eval_path.exists():
        eval_path.unlink()

    total_versions = 0
    compile_ok_versions = 0
    function_ok_versions = 0
    task_compile_pass = 0
    task_function_pass = 0

    with open(eval_path, "w", encoding="utf-8") as handle:
        for record_index, record in enumerate(records, start=1):
            task_id = int(record["id"])
            task = tasks_by_id.get(task_id)
            task_name = record.get("task_name", "Unknown")
            sample_results: Dict[str, Any] = {}
            task_has_compile = False
            task_has_function = False

            if task is None:
                output = {
                    "id": task_id,
                    "task_name": task_name,
                    "compile_pass": False,
                    "functionality_pass": False,
                    "error": "missing dataset task",
                    "eval_mode": "harness_kernel_only",
                    "prompt": record.get("prompt", ""),
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                continue

            print(f"[EVAL {record_index}/{len(records)}] id={task_id} {task_name}")
            for sample_idx, code in iter_code_versions(record, args.num_samples):
                total_versions += 1
                compile_ok = False
                functionality = False
                full_source = ""
                work_dir = temp_root / f"task_{task_id}_sample_{sample_idx}"
                if work_dir.exists():
                    shutil.rmtree(work_dir, ignore_errors=True)
                work_dir.mkdir(parents=True, exist_ok=True)

                if code:
                    full_source = build_harness_source(task, code)
                    with suppress_output(enabled=True):
                        executable_path = compile_code(full_source, str(work_dir), timeout=args.compile_timeout)
                        if executable_path is not None:
                            compile_ok = True
                            functionality = evaluate_functionality(
                                task=task,
                                executable_path=executable_path,
                                work_dir=str(work_dir),
                                timeout=args.run_timeout,
                                run_script_as_function=helpers.run_script_as_function,
                            )

                if args.keep_temp and full_source:
                    (work_dir / "harness_source.cu").write_text(full_source, encoding="utf-8")

                sample_results[f"compile{sample_idx}"] = compile_ok
                sample_results[f"functionality{sample_idx}"] = functionality
                if compile_ok:
                    compile_ok_versions += 1
                    task_has_compile = True
                if functionality:
                    function_ok_versions += 1
                    task_has_function = True

                if not args.keep_temp:
                    shutil.rmtree(work_dir, ignore_errors=True)

            output = {
                "id": task_id,
                "task_name": task_name,
                "compile_pass": task_has_compile,
                "functionality_pass": task_has_function,
                **sample_results,
                "eval_mode": "harness_kernel_only",
                "level": record.get("level", args.level),
                "model_path": record.get("model_path", args.model_path),
                "scem_checkpoint": record.get("scem_checkpoint", args.scem_checkpoint),
                "lora_checkpoint": record.get("lora_checkpoint", args.lora_checkpoint),
                "kernel_signature": record.get("kernel_signature", ""),
                "prompt": record.get("prompt", ""),
            }
            for sample_idx in range(1, args.num_samples + 1):
                code_key = f"code{sample_idx}"
                response_key = f"response{sample_idx}"
                extraction_key = f"extraction_errors{sample_idx}"
                if code_key in record:
                    output[code_key] = record.get(code_key)
                if response_key in record:
                    output[response_key] = record.get(response_key)
                if extraction_key in record:
                    output[extraction_key] = record.get(extraction_key)

            if task_has_compile:
                task_compile_pass += 1
            if task_has_function:
                task_function_pass += 1
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()

    total_tasks = len(records)
    summary = {
        "total_tasks": total_tasks,
        "total_versions": total_versions,
        "output_dir": str(eval_path.parent),
        "eval_mode": "harness_kernel_only",
        "level": args.level,
        "model_path": args.model_path,
        "scem_checkpoint": args.scem_checkpoint,
        "lora_checkpoint": args.lora_checkpoint,
        "num_samples": args.num_samples,
        "start_index": args.start_index,
        "task_stride": args.task_stride,
        "limit": args.limit,
        "compile_accuracy": safe_div(compile_ok_versions, total_versions),
        "functionality_accuracy": safe_div(function_ok_versions, total_versions),
        "task_compile_pass_rate": safe_div(task_compile_pass, total_tasks),
        "task_functionality_pass_rate": safe_div(task_function_pass, total_tasks),
        "compile_ok_versions": compile_ok_versions,
        "function_ok_versions": function_ok_versions,
        "task_compile_pass": task_compile_pass,
        "task_function_pass": task_function_pass,
        "results_jsonl": str(results_path),
        "eval_jsonl": str(eval_path),
    }
    summary_path = eval_path.parent / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def main():
    args = parse_args()
    cudabench_root, dataset_path = resolve_cudabench_paths(args.cudabench_root, args.dataset)
    helpers = load_cudabench_helpers(cudabench_root)
    tasks = select_eval_tasks(
        load_cudabench_tasks(dataset_path, args.start_index, limit=None),
        stride=args.task_stride,
        limit=args.limit,
    )
    tasks_by_id = {int(task["id"]): task for task in tasks}
    output_dir = Path(args.output_dir) if args.output_dir else build_auto_output_dir(args)
    args.output_dir = str(output_dir)
    results_path = Path(args.results_jsonl) if args.results_jsonl else output_dir / "generated_results.jsonl"
    eval_path = output_dir / "eval_results.jsonl"

    if not args.trust_generated:
        generate_results(args, tasks, results_path, helpers)
    elif not results_path.exists():
        raise FileNotFoundError(f"--trust-generated requires an existing results file: {results_path}")

    summary = evaluate_results(args, tasks_by_id, results_path, eval_path, helpers)
    print("\nHarness evaluation summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
