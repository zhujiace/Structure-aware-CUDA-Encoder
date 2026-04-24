import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CUDABENCH_ROOT = PROJECT_ROOT / "external" / "CUDABench"
DEFAULT_DATASET = DEFAULT_CUDABENCH_ROOT / "Datasets" / "CUDABench-Set.jsonl"
QWEN35_GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 20,
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scem import (  # noqa: E402
    CudaProgramStateExtractor,
    SCEMConfig,
    SCEModule,
    TokenizerCudaStateProvider,
    attach_scem_hidden_state_capture,
    build_scem_logits_processor,
)


@dataclass(frozen=True)
class CUDABenchHelpers:
    run_script_as_function: Any
    prompt_template: str
    system_prompt: str


def resolve_cudabench_paths(
    cudabench_root: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Tuple[Path, Path]:
    root = Path(cudabench_root).resolve() if cudabench_root else DEFAULT_CUDABENCH_ROOT.resolve()
    dataset_path = Path(dataset).resolve() if dataset else root / "Datasets" / "CUDABench-Set.jsonl"
    if not root.exists():
        raise FileNotFoundError(
            f"CUDABench root not found: {root}. "
            "Run `git submodule update --init --recursive` or pass --cudabench-root."
        )
    if not dataset_path.exists():
        raise FileNotFoundError(f"CUDABench dataset not found: {dataset_path}")
    return root, dataset_path


def load_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cudabench_helpers(cudabench_root: Path) -> CUDABenchHelpers:
    evaluator_core = load_module_from_path(
        "cudabench_evaluator_core",
        cudabench_root / "Evaluate" / "evaluator_core.py",
    )
    prompt_module = load_module_from_path(
        "cudabench_prompt",
        cudabench_root / "Generate" / "prompt.py",
    )
    return CUDABenchHelpers(
        run_script_as_function=evaluator_core.run_script_as_function,
        prompt_template=prompt_module.PROMPT,
        system_prompt=prompt_module.SYSTEM_PROMPT,
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_cudabench_tasks(
    path: Path,
    start_index: int = 0,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    tasks = load_jsonl(path)
    tasks = tasks[start_index:]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def select_cudabench_tasks(
    tasks: List[Dict[str, Any]],
    task_id: Optional[int] = None,
    task_ids: Optional[str] = None,
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    by_id = {int(task["id"]): task for task in tasks}
    selected_ids: List[int] = []

    if task_id is not None:
        selected_ids.append(task_id)
    if task_ids:
        selected_ids.extend(int(part.strip()) for part in task_ids.split(",") if part.strip())
    if start_id is not None or end_id is not None:
        range_start = start_id if start_id is not None else min(by_id)
        range_end = end_id if end_id is not None else max(by_id)
        selected_ids.extend(
            candidate_id for candidate_id in sorted(by_id) if range_start <= candidate_id <= range_end
        )

    if selected_ids:
        unique_ids: List[int] = []
        seen = set()
        for selected_id in selected_ids:
            if selected_id not in seen:
                seen.add(selected_id)
                unique_ids.append(selected_id)
        return [by_id[selected_id] for selected_id in unique_ids if selected_id in by_id]

    tasks = sorted(tasks, key=lambda item: int(item["id"]))
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def build_cuda_prompt(
    entry: Dict[str, Any],
    level: str,
    gpu_model: str,
    prompt_template: str,
) -> str:
    input_spec = "\n".join(
        f"{item['name']}: {item['dtype']}, shape = {item['shape']}" for item in entry["inputs"]
    )
    output_spec = "\n".join(
        f"{item['name']}: {item['dtype']}, shape = {item['shape']}" for item in entry["outputs"]
    )
    return prompt_template.format(
        task_name=entry["task_name"],
        task_description=entry[level],
        input_spec=input_spec,
        output_spec=output_spec,
        gpu=gpu_model,
    )


def extract_code(response: str) -> str:
    matches = re.findall(r"```(?:\w+)?\s*([\s\S]*?)```", response or "")
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
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int,
        system_prompt: str,
        enable_scem: bool = False,
        scem_checkpoint: Optional[str] = None,
        alpha: float = 0.3,
        task_family: str = "unknown",
        tensor_rank: int = 0,
    ):
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.logits_processor = None

        if enable_scem or scem_checkpoint:
            model_dtype = next(self.model.parameters()).dtype
            if scem_checkpoint:
                scem = load_scem_checkpoint(scem_checkpoint, self.model.config, self.device, model_dtype)
            else:
                scem_config = SCEMConfig.from_lm_config(self.model.config)
                scem = SCEModule(scem_config).to(device=self.device, dtype=model_dtype)
                scem.eval()
            state_provider = TokenizerCudaStateProvider(
                tokenizer=self.tokenizer,
                extractor=CudaProgramStateExtractor(
                    task_family=task_family,
                    tensor_rank=tensor_rank,
                ),
            )
            attach_scem_hidden_state_capture(self.model)
            self.logits_processor = LogitsProcessorList(
                [
                    build_scem_logits_processor(
                        model=self.model,
                        scem=scem,
                        state_provider=state_provider,
                        alpha=alpha,
                    )
                ]
            )

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        generation_kwargs = dict(QWEN35_GENERATION_KWARGS)
        generation_kwargs["max_new_tokens"] = self.max_new_tokens
        if self.logits_processor is not None:
            generation_kwargs["logits_processor"] = self.logits_processor

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def load_done_ids(path: Path) -> set[int]:
    done = set()
    if not path.exists():
        return done
    for item in load_jsonl(path):
        try:
            done.add(int(item["id"]))
        except Exception:
            continue
    return done


def load_generated_results(path: Path) -> List[Dict[str, Any]]:
    return load_jsonl(path)


def iter_code_versions(record: Dict[str, Any], num_samples: int) -> Iterator[Tuple[int, Optional[str]]]:
    if "code" in record:
        yield 1, record.get("code")
    for sample_idx in range(1, num_samples + 1):
        key = f"code{sample_idx}"
        if key in record:
            yield sample_idx, record.get(key)


@contextlib.contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


@contextlib.contextmanager
def temporary_work_dir(base: Path, name: str, keep: bool):
    work_dir = base / name
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield work_dir
    finally:
        if not keep:
            shutil.rmtree(work_dir, ignore_errors=True)


def compile_code(code: str, work_dir: str, timeout: int, code_filename: str = "kernel.cu") -> Optional[str]:
    cu_file_path = os.path.join(work_dir, code_filename)
    executable_path = os.path.join(work_dir, "kernel")
    with open(cu_file_path, "w", encoding="utf-8") as handle:
        handle.write(code)
    try:
        subprocess.run(
            ["nvcc", cu_file_path, "-o", executable_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        return executable_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def run_executable(executable_path: str, work_dir: str, timeout: int) -> bool:
    try:
        result = subprocess.run(
            ["./" + os.path.basename(executable_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            cwd=work_dir,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def evaluate_functionality(
    task: Dict[str, Any],
    executable_path: str,
    work_dir: str,
    timeout: int,
    run_script_as_function,
) -> bool:
    os.makedirs(os.path.join(work_dir, "data"), exist_ok=True)
    ok, _ = run_script_as_function(task.get("gen.py", ""), work_dir=work_dir)
    if not ok:
        return False
    if not run_executable(executable_path, work_dir, timeout):
        return False
    ok, compare_out = run_script_as_function(task.get("compare.py", ""), work_dir=work_dir)
    if not ok:
        return False
    return "F" not in compare_out


def safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
