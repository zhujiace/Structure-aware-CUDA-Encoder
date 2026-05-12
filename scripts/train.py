import argparse
import csv
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scem import CudaProgramStateBatch, CudaProgramStateExtractor, SCEMConfig, SCEModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train SCEM on next-token CUDA generation steps.")
    parser.add_argument("--model-path", default="./models/Qwen3.5-0.8B")
    parser.add_argument("--train-file", required=True, help="JSONL file with text, prompt/completion, or messages.")
    parser.add_argument("--output-dir", default="./checkpoints/scem")
    parser.add_argument("--train-output-dir", default="train_outputs", help="Directory for training logs, metrics, summaries, and optional loss plots.")
    parser.add_argument("--train-run-name", default=None, help="Optional name for the subdirectory under --train-output-dir.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-prefix-length", type=int, default=16)
    parser.add_argument("--skip-overlength", action="store_true", help="Skip examples longer than --max-length instead of truncating them.")
    parser.add_argument("--max-raw-examples", type=int, default=None, help="Read at most this many raw records; useful for smoke tests.")
    parser.add_argument("--max-training-points", type=int, default=None, help="Keep at most this many expanded training points; useful for smoke tests.")
    parser.add_argument(
        "--val-ratio",
        "--var-ratio",
        dest="val_ratio",
        type=float,
        default=0.0,
        help="Fraction of raw records reserved for validation. --var-ratio is accepted as a compatibility alias.",
    )
    parser.add_argument("--region-points-per-example", type=int, default=8)
    parser.add_argument("--random-points-per-example", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--bias-rank", type=int, default=64)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--allow-bitsandbytes-lora", action="store_true", help="Allow PEFT to use bitsandbytes LoRA dispatchers if the local bnb install is known-good.")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--model-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return parser.parse_args()


REGION_ANCHORS = {
    "signature": ("__global__", "__device__", "__host__"),
    "indexing": ("threadIdx.", "blockIdx.", "blockDim.", "int idx", "int tid", "int row", "int col"),
    "guard": ("if (", "if(", "&&", "||"),
    "shared_memory": ("extern __shared__", "__shared__"),
    "synchronization": ("__syncthreads",),
    "write_back": ("] =", "]=", "atomicAdd", "atomicMax", "atomicMin"),
    "statement_close": (";", "}"),
}


@dataclass(frozen=True)
class TrainingPoint:
    ids: List[int]
    target_pos: int
    source: str
    state_prefix: str


class PrefixNextTokenDataset(Dataset):
    """Expand each CUDA example into multiple region-aware next-token steps."""

    def __init__(
        self,
        path: Optional[str],
        tokenizer,
        max_length: int,
        min_prefix_length: int,
        region_points_per_example: int,
        random_points_per_example: int,
        skip_overlength: bool = False,
        max_raw_examples: Optional[int] = None,
        max_training_points: Optional[int] = None,
        records: Optional[Sequence[Dict[str, Any]]] = None,
        split_name: str = "training",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_prefix_length = min_prefix_length
        self.region_points_per_example = region_points_per_example
        self.random_points_per_example = random_points_per_example
        self.examples = []
        raw_examples = 0
        skipped_overlength = 0
        if records is None:
            if path is None:
                raise ValueError("Either path or records must be provided")
            record_iter = self._iter_records(path)
        else:
            record_iter = iter(records)
        for record in record_iter:
            if max_raw_examples is not None and raw_examples >= max_raw_examples:
                break
            text, target_char_start = self._record_to_text(record)
            raw_examples += 1
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=not skip_overlength,
                max_length=max_length if not skip_overlength else None,
            )
            ids = encoded.input_ids
            if skip_overlength and len(ids) > max_length:
                skipped_overlength += 1
                continue
            if len(ids) > min_prefix_length:
                offsets = encoded.offset_mapping
                self.examples.extend(
                    self._build_training_points(
                        text=text,
                        ids=ids,
                        offsets=offsets,
                        target_char_start=target_char_start,
                    )
                )
            if max_training_points is not None and len(self.examples) >= max_training_points:
                self.examples = self.examples[:max_training_points]
                break
        if not self.examples:
            raise ValueError("No usable training examples found")
        message = f"Loaded {len(self.examples)} {split_name} points from {raw_examples} raw examples"
        if skipped_overlength:
            message += f" ({skipped_overlength} overlength examples skipped)"
        print(message)

    @staticmethod
    def load_records(path: str, max_raw_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        records = []
        for record in PrefixNextTokenDataset._iter_records(path):
            if max_raw_examples is not None and len(records) >= max_raw_examples:
                break
            records.append(record)
        return records

    @staticmethod
    def _iter_records(path: str):
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                yield from data
                return
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    yield from data["data"]
                    return
                yield data
                return
            raise ValueError("JSON training file must contain an object or a list of objects")

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _record_to_text(self, record: Dict[str, Any]) -> Tuple[str, int]:
        if "messages" in record:
            text = self.tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            assistant_contents = [
                message.get("content", "")
                for message in record["messages"]
                if message.get("role") == "assistant"
            ]
            if not assistant_contents:
                return text, 0
            target_start = text.rfind(assistant_contents[-1])
            return text, max(0, target_start)
        if "prompt" in record and "completion" in record:
            messages = [{"role": "user", "content": record["prompt"]}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt + record["completion"], len(prompt)
        if "instruction" in record and "output" in record:
            user_content = record["instruction"]
            if record.get("input"):
                user_content = user_content.rstrip() + "\n\n" + record["input"].lstrip()
            messages = [{"role": "user", "content": user_content}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt + record["output"], len(prompt)
        if "text" in record:
            return record["text"], 0
        raise ValueError(
            "Each record must contain text, prompt/completion, messages, or instruction/output"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        point = self.examples[index]
        return {
            "prefix_ids": point.ids[: point.target_pos],
            "label": point.ids[point.target_pos],
            "source": point.source,
            "state_prefix": point.state_prefix,
        }

    def _build_training_points(
        self,
        text: str,
        ids: List[int],
        offsets: Sequence[Tuple[int, int]],
        target_char_start: int,
    ) -> List[TrainingPoint]:
        positions = []
        seen = set()
        min_target_pos = self._char_to_token_pos(target_char_start, offsets) or self.min_prefix_length
        min_target_pos = max(self.min_prefix_length, min_target_pos)
        if min_target_pos >= len(ids):
            return []
        for region, anchors in REGION_ANCHORS.items():
            for anchor in anchors:
                char_pos = text.find(anchor, target_char_start)
                if char_pos < 0:
                    continue
                anchor_end = min(len(text) - 1, char_pos + len(anchor))
                token_pos = self._char_to_token_pos(anchor_end, offsets)
                if self._is_valid_target(token_pos, ids, min_target_pos) and token_pos not in seen:
                    positions.append((token_pos, region))
                    seen.add(token_pos)
                break

        positions = positions[: self.region_points_per_example]

        random_budget = self.random_points_per_example
        if random_budget > 0:
            valid_range = range(min_target_pos, len(ids))
            random_positions = random.sample(
                list(valid_range),
                k=min(random_budget, max(0, len(ids) - min_target_pos)),
            )
            for token_pos in random_positions:
                if token_pos not in seen:
                    positions.append((token_pos, "random"))
                    seen.add(token_pos)

        if not positions:
            fallback = random.randint(min_target_pos, len(ids) - 1)
            positions.append((fallback, "fallback"))

        points = []
        for token_pos, source in positions:
            prefix_end = offsets[token_pos][0] if token_pos < len(offsets) else len(text)
            prefix_end = max(target_char_start, prefix_end)
            state_prefix = text[target_char_start:prefix_end]
            points.append(
                TrainingPoint(
                    ids=ids,
                    target_pos=token_pos,
                    source=source,
                    state_prefix=state_prefix,
                )
            )
        return points

    def _char_to_token_pos(
        self,
        char_pos: int,
        offsets: Sequence[Tuple[int, int]],
    ) -> Optional[int]:
        for index, (start, end) in enumerate(offsets):
            if start == end:
                continue
            if start <= char_pos < end or char_pos < end:
                return index
        return None

    def _is_valid_target(self, token_pos: Optional[int], ids: List[int], min_target_pos: int) -> bool:
        if token_pos is None:
            return False
        return min_target_pos <= token_pos < len(ids)


@dataclass
class PrefixBatch:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor
    state_prefix_texts: List[str]


class PrefixCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, examples: List[Dict[str, Any]]) -> PrefixBatch:
        sequences = [example["prefix_ids"] for example in examples]
        labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
        max_len = max(len(sequence) for sequence in sequences)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, -length:] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, -length:] = 1
        return PrefixBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            state_prefix_texts=[example["state_prefix"] for example in examples],
        )


def configure_backbone(model, args):
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        if not args.allow_bitsandbytes_lora:
            import peft.import_utils as peft_import_utils
            import peft.tuners.lora.model as peft_lora_model

            peft_import_utils.is_bnb_available = lambda: False
            peft_import_utils.is_bnb_4bit_available = lambda: False
            peft_lora_model.is_bnb_available = lambda: False
            peft_lora_model.is_bnb_4bit_available = lambda: False

        if args.freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.print_trainable_parameters()
    return model


def get_lm_config(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model().config
    return model.config


def resolve_model_dtype(args, accelerator: Accelerator) -> torch.dtype:
    if accelerator.device.type != "cuda":
        return torch.float32
    if args.model_dtype == "float32":
        return torch.float32
    if args.model_dtype == "float16":
        return torch.float16
    if args.model_dtype == "bfloat16":
        return torch.bfloat16
    if accelerator.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def save_checkpoint(
    output_dir: str,
    scem: SCEModule,
    model,
    tokenizer,
    args,
    step: int,
    accelerator: Optional[Accelerator] = None,
    checkpoint_name: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
):
    path = os.path.join(output_dir, checkpoint_name or f"step-{step}")
    os.makedirs(path, exist_ok=True)
    scem_to_save = accelerator.unwrap_model(scem) if accelerator is not None else scem
    model_to_save = accelerator.unwrap_model(model) if accelerator is not None else model
    torch.save({"config": scem_to_save.config, "state_dict": scem_to_save.state_dict()}, os.path.join(path, "scem.pt"))
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, "training_args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    if metrics is not None:
        with open(os.path.join(path, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
    if args.use_lora and hasattr(model_to_save, "save_pretrained"):
        model_to_save.save_pretrained(os.path.join(path, "lora"))


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "run"


class TrainingRunLogger:
    fieldnames = [
        "event",
        "epoch",
        "step",
        "train_loss",
        "val_loss",
        "lr",
        "best_val_loss",
        "best_step",
    ]

    def __init__(self, args, train_points: int, val_points: int, total_steps: int):
        base = Path(args.train_output_dir)
        run_name = args.train_run_name or f"{slugify(Path(args.output_dir).name)}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        self.path = base / run_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.path / "metrics.jsonl"
        self.csv_path = self.path / "metrics.csv"
        self.summary_path = self.path / "summary.json"
        self.figs_dir = self.path / "figs"
        self.figs_dir.mkdir(parents=True, exist_ok=True)
        self.train_plot_path = self.figs_dir / "train_loss.png"
        self.val_plot_path = self.figs_dir / "val_loss.png"
        self.records: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {
            "run_name": run_name,
            "output_dir": args.output_dir,
            "train_output_dir": str(self.path),
            "train_points": train_points,
            "val_points": val_points,
            "total_steps": total_steps,
            "best_step": None,
            "best_val_loss": None,
            "final_step": None,
            "final_val_loss": None,
            "figs_dir": str(self.figs_dir),
            "train_loss_plot": str(self.train_plot_path),
            "val_loss_plot": str(self.val_plot_path),
        }
        with open(self.path / "training_args.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
        self._write_summary()

    def append(self, update_plot: bool = False, **record) -> None:
        normalized = {key: record.get(key) for key in self.fieldnames}
        self.records.append(normalized)
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(normalized)
        if normalized.get("val_loss") is not None:
            self.summary["final_val_loss"] = normalized["val_loss"]
        if normalized.get("best_val_loss") is not None:
            self.summary["best_val_loss"] = normalized["best_val_loss"]
            self.summary["best_step"] = normalized.get("best_step")
        self.summary["final_step"] = normalized.get("step")
        self._write_summary()
        if update_plot:
            self._write_plot()

    def finish(self, final_step: int, best_step: int, best_val_loss: float, final_val_loss: Optional[float]) -> None:
        self.summary["final_step"] = final_step
        if best_step > 0 and math.isfinite(best_val_loss):
            self.summary["best_step"] = best_step
            self.summary["best_val_loss"] = best_val_loss
        if final_val_loss is not None:
            self.summary["final_val_loss"] = final_val_loss
        self._write_summary()
        self._write_plot()

    def _write_summary(self) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(self.summary, handle, ensure_ascii=False, indent=2)

    def _write_plot(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        train_steps = [row["step"] for row in self.records if row.get("event") == "train" and row.get("train_loss") is not None]
        train_losses = [row["train_loss"] for row in self.records if row.get("event") == "train" and row.get("train_loss") is not None]
        val_steps = [row["step"] for row in self.records if row.get("val_loss") is not None]
        val_losses = [row["val_loss"] for row in self.records if row.get("val_loss") is not None]
        if not train_steps and not val_steps:
            return

        if train_steps:
            plt.figure(figsize=(8, 4.5))
            plt.plot(train_steps, train_losses, label="train_loss")
            plt.xlabel("step")
            plt.ylabel("train loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.train_plot_path)
            plt.close()
        if val_steps:
            plt.figure(figsize=(8, 4.5))
            plt.plot(val_steps, val_losses, marker="o", label="val_loss")
            best_step = self.summary.get("best_step")
            best_val_loss = self.summary.get("best_val_loss")
            if best_step is not None and best_val_loss is not None:
                plt.scatter([best_step], [best_val_loss], color="red", zorder=3, label=f"best step {best_step}")
            plt.xlabel("step")
            plt.ylabel("validation loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.val_plot_path)
            plt.close()


def split_train_validation_records(
    records: Sequence[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    records = list(records)
    if val_ratio == 0.0:
        return records, []
    if len(records) < 2:
        raise ValueError("--val-ratio requires at least two raw records")

    val_count = max(1, int(round(len(records) * val_ratio)))
    val_count = min(val_count, len(records) - 1)
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_indices = set(indices[:val_count])
    train_records = [record for index, record in enumerate(records) if index not in val_indices]
    val_records = [record for index, record in enumerate(records) if index in val_indices]
    return train_records, val_records


def compute_scem_loss(model, scem, batch: PrefixBatch, extractor: CudaProgramStateExtractor, args, accelerator: Accelerator):
    input_ids = batch.input_ids.to(accelerator.device)
    attention_mask = batch.attention_mask.to(accelerator.device)
    labels = batch.labels.to(accelerator.device)
    states = extractor.extract_batch(batch.state_prefix_texts)
    state_batch = CudaProgramStateBatch.from_states(states, device=accelerator.device)

    if args.use_lora:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    else:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
    last_hidden = outputs.hidden_states[-1][:, -1, :]
    last_logits = outputs.logits[:, -1, :]
    scem_bias = scem(last_hidden, state_batch).bias.to(dtype=last_logits.dtype)
    adjusted_logits = last_logits + args.alpha * scem_bias
    return F.cross_entropy(adjusted_logits, labels)


@torch.no_grad()
def evaluate_validation(model, scem, dataloader, extractor: CudaProgramStateExtractor, args, accelerator: Accelerator) -> float:
    if dataloader is None:
        raise ValueError("Validation dataloader is not available")

    model_was_training = model.training
    scem_was_training = scem.training
    model.eval()
    scem.eval()

    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_count = torch.tensor(0.0, device=accelerator.device)
    for batch in dataloader:
        with accelerator.autocast():
            loss = compute_scem_loss(model, scem, batch, extractor, args, accelerator)
        count = torch.tensor(batch.labels.numel(), dtype=torch.float32, device=accelerator.device)
        total_loss += loss.detach().float() * count
        total_count += count

    stats = torch.stack([total_loss, total_count])
    gathered = accelerator.gather(stats).view(-1, 2)
    total_loss_value = gathered[:, 0].sum().item()
    total_count_value = gathered[:, 1].sum().item()
    val_loss = total_loss_value / max(total_count_value, 1.0)

    if model_was_training:
        model.train()
    if scem_was_training:
        scem.train()
    return val_loss


def main():
    args = parse_args()
    mixed_precision = args.mixed_precision if torch.cuda.is_available() else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=mixed_precision,
    )
    random.seed(args.seed + accelerator.process_index)
    set_seed(args.seed)

    dtype = resolve_model_dtype(args, accelerator)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype)
    model = configure_backbone(model, args)
    model.train(args.use_lora)
    if not args.use_lora:
        model.eval()

    scem_config = SCEMConfig.from_lm_config(get_lm_config(model), bias_rank=args.bias_rank)
    scem = SCEModule(scem_config).to(dtype=next(model.parameters()).dtype)
    scem.train()

    raw_records = PrefixNextTokenDataset.load_records(args.train_file, max_raw_examples=args.max_raw_examples)
    train_records, val_records = split_train_validation_records(
        raw_records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if accelerator.is_main_process and val_records:
        print(f"Validation split: {len(val_records)} raw examples; training split: {len(train_records)} raw examples")

    dataset = PrefixNextTokenDataset(
        path=None,
        tokenizer=tokenizer,
        max_length=args.max_length,
        min_prefix_length=args.min_prefix_length,
        region_points_per_example=args.region_points_per_example,
        random_points_per_example=args.random_points_per_example,
        skip_overlength=args.skip_overlength,
        max_training_points=args.max_training_points,
        records=train_records,
        split_name="training",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PrefixCollator(tokenizer),
        num_workers=0,
    )
    val_dataloader = None
    if val_records:
        val_dataset = PrefixNextTokenDataset(
            path=None,
            tokenizer=tokenizer,
            max_length=args.max_length,
            min_prefix_length=args.min_prefix_length,
            region_points_per_example=args.region_points_per_example,
            random_points_per_example=args.random_points_per_example,
            skip_overlength=args.skip_overlength,
            max_training_points=None,
            records=val_records,
            split_name="validation",
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=PrefixCollator(tokenizer),
            num_workers=0,
        )

    trainable_params = list(scem.parameters()) + [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(dataloader) * args.epochs / (args.grad_accum_steps * accelerator.num_processes))
    total_steps = max(1, total_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    extractor = CudaProgramStateExtractor()
    train_logger = None
    if accelerator.is_main_process:
        train_logger = TrainingRunLogger(
            args=args,
            train_points=len(dataset),
            val_points=len(val_dataset) if val_dataloader is not None else 0,
            total_steps=total_steps,
        )
        print(f"Training metrics will be written to {train_logger.path}")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if val_dataloader is None:
        model, scem, optimizer, dataloader, scheduler = accelerator.prepare(
            model,
            scem,
            optimizer,
            dataloader,
            scheduler,
        )
    else:
        model, scem, optimizer, dataloader, scheduler, val_dataloader = accelerator.prepare(
            model,
            scem,
            optimizer,
            dataloader,
            scheduler,
            val_dataloader,
        )
    trainable_params = [p for p in list(scem.parameters()) + list(model.parameters()) if p.requires_grad]

    global_step = 0
    best_val_loss = float("inf")
    best_step = 0
    final_val_loss = None
    last_train_loss = None
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for local_step, batch in enumerate(dataloader):
            with accelerator.accumulate(model, scem):
                with accelerator.autocast():
                    loss = compute_scem_loss(model, scem, batch, extractor, args, accelerator)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.log_steps == 0:
                    current_loss = accelerator.gather(loss.detach()).mean().item()
                    last_train_loss = current_loss
                    lr = scheduler.get_last_lr()[0]
                    accelerator.print(f"epoch={epoch} step={global_step} loss={current_loss:.4f} lr={lr:.2e}")
                    if train_logger is not None:
                        train_logger.append(
                            event="train",
                            epoch=epoch,
                            step=global_step,
                            train_loss=current_loss,
                            lr=lr,
                            best_val_loss=best_val_loss if math.isfinite(best_val_loss) else None,
                            best_step=best_step or None,
                        )

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    metrics = {"step": global_step}
                    if last_train_loss is None:
                        last_train_loss = accelerator.gather(loss.detach()).mean().item()
                    if val_dataloader is not None:
                        val_loss = evaluate_validation(model, scem, val_dataloader, extractor, args, accelerator)
                        metrics["val_loss"] = val_loss
                        accelerator.print(f"epoch={epoch} step={global_step} val_loss={val_loss:.4f}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = global_step
                            if accelerator.is_main_process:
                                save_checkpoint(
                                    args.output_dir,
                                    scem,
                                    model,
                                    tokenizer,
                                    args,
                                    global_step,
                                    accelerator,
                                    checkpoint_name="best",
                                    metrics={**metrics, "best_step": best_step},
                                )
                            accelerator.wait_for_everyone()
                        if train_logger is not None:
                            train_logger.append(
                                event="validation",
                                epoch=epoch,
                                step=global_step,
                                train_loss=last_train_loss,
                                val_loss=val_loss,
                                lr=scheduler.get_last_lr()[0],
                                best_val_loss=best_val_loss,
                                best_step=best_step,
                                update_plot=True,
                            )
                    if accelerator.is_main_process:
                        save_checkpoint(
                            args.output_dir,
                            scem,
                            model,
                            tokenizer,
                            args,
                            global_step,
                            accelerator,
                            metrics=metrics,
                        )
                    accelerator.wait_for_everyone()

    final_metrics = {"step": global_step}
    if val_dataloader is not None:
        val_loss = evaluate_validation(model, scem, val_dataloader, extractor, args, accelerator)
        final_val_loss = val_loss
        final_metrics["val_loss"] = val_loss
        accelerator.print(f"final step={global_step} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = global_step
            if accelerator.is_main_process:
                save_checkpoint(
                    args.output_dir,
                    scem,
                    model,
                    tokenizer,
                    args,
                    global_step,
                    accelerator,
                    checkpoint_name="best",
                    metrics={**final_metrics, "best_step": best_step},
                )
            accelerator.wait_for_everyone()
        final_metrics["best_val_loss"] = best_val_loss
        final_metrics["best_step"] = best_step
    if accelerator.is_main_process:
        if train_logger is not None:
            train_logger.append(
                event="final",
                epoch=args.epochs,
                step=global_step,
                train_loss=last_train_loss,
                val_loss=final_val_loss,
                lr=scheduler.get_last_lr()[0],
                best_val_loss=best_val_loss if math.isfinite(best_val_loss) else None,
                best_step=best_step or None,
                update_plot=True,
            )
            train_logger.finish(
                final_step=global_step,
                best_step=best_step,
                best_val_loss=best_val_loss,
                final_val_loss=final_val_loss,
            )
        save_checkpoint(args.output_dir, scem, model, tokenizer, args, global_step, accelerator, metrics=final_metrics)
        save_checkpoint(
            args.output_dir,
            scem,
            model,
            tokenizer,
            args,
            global_step,
            accelerator,
            checkpoint_name="final",
            metrics=final_metrics,
        )
        print(f"Training complete. Saved final checkpoint to {args.output_dir}/final and {args.output_dir}/step-{global_step}")
    accelerator.wait_for_everyone()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
