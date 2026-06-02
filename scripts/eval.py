import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

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
    parser.add_argument("--lora-checkpoint", default=None, help="Optional path to a PEFT/LoRA adapter directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Evaluation output directory. If omitted, a unique directory is created under "
            "./eval_outputs/<model>/<YYMMDD>/ using model, level, and mode."
        ),
    )
    parser.add_argument("--run-name", default=None, help="Optional label included in the auto-generated output directory name.")
    parser.add_argument("--level", choices=["level1_prompt", "level2_prompt", "level3_prompt"], default="level3_prompt")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N tasks.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--task-stride",
        type=int,
        default=1,
        help="Evaluate every Nth task after --start-index. Use 5 to sample one kernel from each five-task group.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--use-scem-prompt", action="store_true", help="Enable SCEM-side supplemental system constraints without modifying external/CUDABench.")
    parser.add_argument("--compile-timeout", type=int, default=60)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--trust-generated", action="store_true", help="Skip generation and evaluate existing results.jsonl.")
    parser.add_argument("--results-jsonl", default=None, help="Existing generated results JSONL for --trust-generated.")
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "run"


def checkpoint_label(path: str | None) -> str:
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
    mode_parts = ["baseline"]
    if args.use_scem_prompt:
        mode_parts.append("scemprompt")
    if args.scem_checkpoint:
        mode_parts.append("scem")
        mode_parts.append(slugify(checkpoint_label(args.scem_checkpoint)))
    if args.lora_checkpoint:
        mode_parts.append("lora")
        mode_parts.append(slugify(checkpoint_label(args.lora_checkpoint)))
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
    name = "_".join([level_label, *mode_parts])
    return uniquify_output_dir(Path("eval_outputs") / model_label / date_label / name)


def select_eval_tasks(tasks: List[Dict[str, Any]], stride: int, limit: int | None) -> List[Dict[str, Any]]:
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
        system_prompt=helpers.system_prompt,
        use_scem_prompt=args.use_scem_prompt,
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

            prompt = build_cuda_prompt(task, args.level, args.gpu_model, helpers.prompt_template)
            record = {
                "id": task_id,
                "task_name": task["task_name"],
                "prompt": prompt,
                "level": args.level,
                "model_path": args.model_path,
                "scem_checkpoint": args.scem_checkpoint,
                "lora_checkpoint": args.lora_checkpoint,
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
    first_sample_compile_pass = 0
    first_sample_function_pass = 0

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
                    "level": record.get("level", args.level),
                    "model_path": record.get("model_path", args.model_path),
                    "scem_checkpoint": record.get("scem_checkpoint", args.scem_checkpoint),
                    "lora_checkpoint": record.get("lora_checkpoint", args.lora_checkpoint),
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
                if sample_idx == 1:
                    if compile_ok:
                        first_sample_compile_pass += 1
                    if functionality:
                        first_sample_function_pass += 1

                if not args.keep_temp:
                    shutil.rmtree(work_dir, ignore_errors=True)

            output = {
                "id": task_id,
                "task_name": task_name,
                "compile_pass": task_has_compile,
                "functionality_pass": task_has_function,
                **sample_results,
                "level": record.get("level", args.level),
                "model_path": record.get("model_path", args.model_path),
                "scem_checkpoint": record.get("scem_checkpoint", args.scem_checkpoint),
                "lora_checkpoint": record.get("lora_checkpoint", args.lora_checkpoint),
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
        "output_dir": str(eval_path.parent),
        "level": args.level,
        "model_path": args.model_path,
        "scem_checkpoint": args.scem_checkpoint,
        "lora_checkpoint": args.lora_checkpoint,
        "use_scem_prompt": args.use_scem_prompt,
        "num_samples": args.num_samples,
        "start_index": args.start_index,
        "task_stride": args.task_stride,
        "limit": args.limit,
        "sample_compile_accuracy": safe_div(compile_ok_versions, total_versions),
        "sample_functionality_accuracy": safe_div(function_ok_versions, total_versions),
        "compile_pass@1": safe_div(first_sample_compile_pass, total_tasks),
        "functionality_pass@1": safe_div(first_sample_function_pass, total_tasks),
        f"compile_pass@{args.num_samples}": safe_div(task_compile_pass, total_tasks),
        f"functionality_pass@{args.num_samples}": safe_div(task_function_pass, total_tasks),
        "sample_compile_pass": compile_ok_versions,
        "sample_functionality_pass": function_ok_versions,
        "compile_pass@1_count": first_sample_compile_pass,
        "functionality_pass@1_count": first_sample_function_pass,
        f"compile_pass@{args.num_samples}_count": task_compile_pass,
        f"functionality_pass@{args.num_samples}_count": task_function_pass,
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
    print("\nEvaluation summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
