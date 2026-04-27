import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from utils import (
    DEFAULT_CUDABENCH_ROOT,
    LocalGenerator,
    build_cuda_prompt,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate Qwen/SCEM on CUDABench compile and functionality accuracy."
    )
    parser.add_argument("--cudabench-root", default=str(DEFAULT_CUDABENCH_ROOT))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model-path", default="./models/Qwen3.5-0.8B")
    parser.add_argument("--scem-checkpoint", default=None, help="Optional path to scem.pt. Omit for backbone baseline.")
    parser.add_argument("--output-dir", default="./eval_outputs/cudabench_qwen35_baseline")
    parser.add_argument("--level", choices=["level1_prompt", "level2_prompt", "level3_prompt"], default="level3_prompt")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N tasks.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--use-scem-prompt", action="store_true", help="Append SCEM-side prompt constraints without modifying external/CUDABench.")
    parser.add_argument("--task-family", default="unknown")
    parser.add_argument("--tensor-rank", type=int, default=0)
    parser.add_argument("--compile-timeout", type=int, default=60)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--trust-generated", action="store_true", help="Skip generation and evaluate existing results.jsonl.")
    parser.add_argument("--results-jsonl", default=None, help="Existing generated results JSONL for --trust-generated.")
    return parser.parse_args()


def generate_results(args, tasks: List[Dict[str, Any]], output_path: Path, helpers) -> None:
    generator = LocalGenerator(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        system_prompt=helpers.system_prompt,
        use_scem_prompt=args.use_scem_prompt,
        enable_scem=bool(args.scem_checkpoint),
        scem_checkpoint=args.scem_checkpoint,
        alpha=args.alpha,
        task_family=args.task_family,
        tensor_rank=args.tensor_rank,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path)

    with open(output_path, "a", encoding="utf-8") as handle:
        for index, task in enumerate(tasks, start=1):
            task_id = int(task["id"])
            if task_id in done_ids:
                print(f"[GEN {index}/{len(tasks)}] id={task_id} skip existing")
                continue

            prompt = build_cuda_prompt(task, args.level, args.gpu_model, helpers.prompt_template)
            record = {
                "id": task_id,
                "task_name": task["task_name"],
                "prompt": prompt,
                "level": args.level,
                "model_path": args.model_path,
                "scem_checkpoint": args.scem_checkpoint,
            }
            print(f"[GEN {index}/{len(tasks)}] id={task_id} {task['task_name']}")
            for sample_idx in range(1, args.num_samples + 1):
                response = generator.generate(prompt)
                code = extract_code(response)
                record[f"response{sample_idx}"] = response
                record[f"code{sample_idx}"] = code
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
            sample_results = {}
            task_has_compile = False
            task_has_function = False

            if task is None:
                output = {
                    "id": task_id,
                    "task_name": task_name,
                    "compile_pass": False,
                    "functionality_pass": False,
                    "error": "missing dataset task",
                    "prompt": record.get("prompt", ""),
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                continue

            print(f"[EVAL {record_index}/{len(records)}] id={task_id} {task_name}")
            for sample_idx, code in iter_code_versions(record, args.num_samples):
                total_versions += 1
                compile_ok = False
                functionality = False
                work_dir = temp_root / f"task_{task_id}_sample_{sample_idx}"
                if work_dir.exists():
                    shutil.rmtree(work_dir, ignore_errors=True)
                work_dir.mkdir(parents=True, exist_ok=True)

                if code:
                    with suppress_output(enabled=True):
                        executable_path = compile_code(code, str(work_dir), timeout=args.compile_timeout)
                        if executable_path is not None:
                            compile_ok = True
                            functionality = evaluate_functionality(
                                task=task,
                                executable_path=executable_path,
                                work_dir=str(work_dir),
                                timeout=args.run_timeout,
                                run_script_as_function=helpers.run_script_as_function,
                            )

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
                "prompt": record.get("prompt", ""),
            }
            for sample_idx in range(1, args.num_samples + 1):
                response_key = f"response{sample_idx}"
                code_key = f"code{sample_idx}"
                if code_key in record:
                    output[code_key] = record.get(code_key)
                if response_key in record:
                    output[response_key] = record.get(response_key)

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
    tasks = load_cudabench_tasks(dataset_path, args.start_index, args.limit)
    tasks_by_id = {int(task["id"]): task for task in tasks}
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_jsonl) if args.results_jsonl else output_dir / "generated_results.jsonl"
    eval_path = output_dir / "eval_results.jsonl"

    if not args.trust_generated:
        generate_results(args, tasks, results_path, helpers)
    elif not results_path.exists():
        raise FileNotFoundError(f"--trust-generated requires an existing results file: {results_path}")

    summary = evaluate_results(args, tasks_by_id, results_path, eval_path, helpers)
    print("\nEvaluation summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
