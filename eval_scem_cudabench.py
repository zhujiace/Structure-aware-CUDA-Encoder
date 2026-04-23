import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from scem import (
    CudaProgramStateExtractor,
    SCEMConfig,
    SCEModule,
    TokenizerCudaStateProvider,
    attach_scem_hidden_state_capture,
    build_scem_logits_processor,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CUDABENCH_ROOT = PROJECT_ROOT / "external" / "CUDABench"
DEFAULT_DATASET = DEFAULT_CUDABENCH_ROOT / "Datasets" / "CUDABench-Set.jsonl"
RUN_SCRIPT_AS_FUNCTION = None
PROMPT_TEMPLATE = None
SYSTEM_PROMPT_TEXT = None


QWEN35_GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 20,
}


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
    parser.add_argument("--task-family", default="unknown")
    parser.add_argument("--tensor-rank", type=int, default=0)
    parser.add_argument("--compile-timeout", type=int, default=60)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--trust-generated", action="store_true", help="Skip generation and evaluate existing results.jsonl.")
    parser.add_argument("--results-jsonl", default=None, help="Existing generated results JSONL for --trust-generated.")
    return parser.parse_args()


def resolve_cudabench_paths(args):
    cudabench_root = Path(args.cudabench_root).resolve()
    dataset = Path(args.dataset).resolve() if args.dataset else cudabench_root / "Datasets" / "CUDABench-Set.jsonl"
    if not cudabench_root.exists():
        raise FileNotFoundError(
            f"CUDABench root not found: {cudabench_root}. "
            "Run `git submodule update --init --recursive` or pass --cudabench-root."
        )
    if not dataset.exists():
        raise FileNotFoundError(f"CUDABench dataset not found: {dataset}")
    return cudabench_root, dataset


def load_cudabench_helpers(cudabench_root: Path):
    evaluator_core = load_module_from_path(
        "cudabench_evaluator_core",
        cudabench_root / "Evaluate" / "evaluator_core.py",
    )
    prompt_module = load_module_from_path(
        "cudabench_prompt",
        cudabench_root / "Generate" / "prompt.py",
    )
    return evaluator_core.run_script_as_function, prompt_module.PROMPT, prompt_module.SYSTEM_PROMPT


def load_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required CUDABench file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset(path: str, start_index: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    tasks = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(json.loads(line))
    tasks = tasks[start_index:]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def build_cuda_prompt(entry: Dict[str, Any], level: str, gpu_model: str) -> str:
    input_spec = "\n".join(
        f"{item['name']}: {item['dtype']}, shape = {item['shape']}" for item in entry["inputs"]
    )
    output_spec = "\n".join(
        f"{item['name']}: {item['dtype']}, shape = {item['shape']}" for item in entry["outputs"]
    )
    return PROMPT_TEMPLATE.format(
        task_name=entry["task_name"],
        task_description=entry[level],
        input_spec=input_spec,
        output_spec=output_spec,
        gpu=gpu_model,
    )


def extract_code(response: str) -> str:
    pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, response or "")
    if matches:
        return matches[0].strip()
    return (response or "").strip()


def load_scem_checkpoint(path: str, model_config, device: str, dtype: torch.dtype) -> SCEModule:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, SCEMConfig):
        config = SCEMConfig.from_lm_config(model_config)
    scem = SCEModule(config)
    scem.load_state_dict(checkpoint["state_dict"])
    scem.to(device=device, dtype=dtype)
    scem.eval()
    return scem


class LocalGenerator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.logits_processor = None

        if args.scem_checkpoint:
            model_dtype = next(self.model.parameters()).dtype
            scem = load_scem_checkpoint(args.scem_checkpoint, self.model.config, self.device, model_dtype)
            state_provider = TokenizerCudaStateProvider(
                tokenizer=self.tokenizer,
                extractor=CudaProgramStateExtractor(
                    task_family=args.task_family,
                    tensor_rank=args.tensor_rank,
                ),
            )
            attach_scem_hidden_state_capture(self.model)
            self.logits_processor = LogitsProcessorList(
                [
                    build_scem_logits_processor(
                        model=self.model,
                        scem=scem,
                        state_provider=state_provider,
                        alpha=args.alpha,
                    )
                ]
            )

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEXT},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        generation_kwargs = dict(QWEN35_GENERATION_KWARGS)
        generation_kwargs["max_new_tokens"] = self.args.max_new_tokens
        if self.logits_processor is not None:
            generation_kwargs["logits_processor"] = self.logits_processor

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_results(args, tasks: List[Dict[str, Any]], output_path: Path) -> None:
    generator = LocalGenerator(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path)

    with open(output_path, "a", encoding="utf-8") as handle:
        for index, task in enumerate(tasks, start=1):
            task_id = int(task["id"])
            if task_id in done_ids:
                print(f"[GEN {index}/{len(tasks)}] id={task_id} skip existing")
                continue

            prompt = build_cuda_prompt(task, args.level, args.gpu_model)
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


def load_done_ids(path: Path) -> set[int]:
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                done.add(int(item["id"]))
            except Exception:
                continue
    return done


def load_generated_results(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def iter_code_versions(record: Dict[str, Any], num_samples: int):
    if "code" in record:
        yield 1, record.get("code")
    for sample_idx in range(1, num_samples + 1):
        key = f"code{sample_idx}"
        if key in record:
            yield sample_idx, record.get(key)


@contextlib.contextmanager
def suppress_eval_output(enabled: bool):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def evaluate_results(args, tasks_by_id: Dict[int, Dict[str, Any]], results_path: Path, eval_path: Path) -> Dict[str, Any]:
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
            output = {
                "id": task_id,
                "task_name": record.get("task_name", "Unknown"),
            }
            task_has_compile = False
            task_has_function = False

            if task is None:
                output["error"] = "missing dataset task"
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                continue

            print(f"[EVAL {record_index}/{len(records)}] id={task_id} {output['task_name']}")
            for sample_idx, code in iter_code_versions(record, args.num_samples):
                total_versions += 1
                compile_ok = False
                functionality = False
                work_dir = temp_root / f"task_{task_id}_sample_{sample_idx}"
                if work_dir.exists():
                    shutil.rmtree(work_dir, ignore_errors=True)
                work_dir.mkdir(parents=True, exist_ok=True)

                if code:
                    with suppress_eval_output(enabled=True):
                        executable_path = compile_code(code, str(work_dir), timeout=args.compile_timeout)
                        if executable_path is not None:
                            compile_ok = True
                            functionality = evaluate_functionality(
                                task,
                                executable_path,
                                str(work_dir),
                                timeout=args.run_timeout,
                            )

                output[f"compile{sample_idx}"] = compile_ok
                output[f"functionality{sample_idx}"] = functionality
                if compile_ok:
                    compile_ok_versions += 1
                    task_has_compile = True
                if functionality:
                    function_ok_versions += 1
                    task_has_function = True

                if not args.keep_temp:
                    shutil.rmtree(work_dir, ignore_errors=True)

            output["compile_pass"] = task_has_compile
            output["functionality_pass"] = task_has_function
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


def safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compile_code(code: str, work_dir: str, timeout: int, code_filename: str = "kernel.cu") -> Optional[str]:
    cu_file_path = os.path.join(work_dir, code_filename)
    executable_path = os.path.join(work_dir, "kernel")
    with open(cu_file_path, "w", encoding="utf-8") as handle:
        handle.write(code)

    command = ["nvcc", cu_file_path, "-o", executable_path]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        return executable_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def evaluate_functionality(
    dataset_task: Dict[str, Any],
    executable_path: str,
    work_dir: str,
    timeout: int,
) -> bool:
    os.makedirs(os.path.join(work_dir, "data"), exist_ok=True)

    ok, _ = RUN_SCRIPT_AS_FUNCTION(dataset_task.get("gen.py", ""), work_dir=work_dir)
    if not ok:
        return False

    if not run_executable(executable_path, work_dir, timeout):
        return False

    ok, compare_out = RUN_SCRIPT_AS_FUNCTION(dataset_task.get("compare.py", ""), work_dir=work_dir)
    if not ok:
        return False
    return "F" not in compare_out


def run_executable(executable_path: str, work_dir: str, timeout: int) -> bool:
    exe_name = "./" + os.path.basename(executable_path)
    try:
        result = subprocess.run(
            [exe_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            cwd=work_dir,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    global RUN_SCRIPT_AS_FUNCTION, PROMPT_TEMPLATE, SYSTEM_PROMPT_TEXT
    args = parse_args()
    cudabench_root, dataset_path = resolve_cudabench_paths(args)
    RUN_SCRIPT_AS_FUNCTION, PROMPT_TEMPLATE, SYSTEM_PROMPT_TEXT = load_cudabench_helpers(cudabench_root)
    tasks = load_dataset(str(dataset_path), args.start_index, args.limit)
    tasks_by_id = {int(task["id"]): task for task in tasks}
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_jsonl) if args.results_jsonl else output_dir / "generated_results.jsonl"
    eval_path = output_dir / "eval_results.jsonl"

    if not args.trust_generated:
        generate_results(args, tasks, results_path)
    elif not results_path.exists():
        raise FileNotFoundError(f"--trust-generated requires an existing results file: {results_path}")

    summary = evaluate_results(args, tasks_by_id, results_path, eval_path)
    print("\nEvaluation summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
