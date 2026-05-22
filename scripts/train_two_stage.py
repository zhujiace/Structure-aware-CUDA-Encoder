import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
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
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scem import CudaASTGraphExtractor, SCEModule
from scem.states import GENERATED_PREFIX_MARKER
from train import (  # noqa: E402
    REGION_ANCHORS,
    TrainingRunLogger,
    build_ast_extractor,
    build_scem_config,
    configure_backbone,
    compute_state_conditioned_loss,
    get_lm_config,
    resolve_model_dtype,
    save_checkpoint,
    split_train_validation_records,
)


@dataclass(frozen=True)
class MultiPointExample:
    ids: List[int]
    target_positions: List[int]
    labels: List[int]
    state_prefix_texts: List[str]


@dataclass
class MultiPointBatch:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    point_batch_indices: torch.LongTensor
    point_positions: torch.LongTensor
    labels: torch.LongTensor
    state_prefix_texts: List[str]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage SCEM training: dense structural warmup followed by harness adaptation."
    )
    parser.add_argument("--model-path", default="./models/Qwen3.5-0.8B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", default="./checkpoints/scem_two_stage")
    parser.add_argument("--train-output-dir", default="train_outputs")
    parser.add_argument("--train-run-name", default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-prefix-length", type=int, default=16)
    parser.add_argument("--skip-overlength", action="store_true")
    parser.add_argument("--max-raw-examples", type=int, default=None)
    parser.add_argument("--max-training-points", type=int, default=None)
    parser.add_argument("--val-ratio", "--var-ratio", dest="val_ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1, help="Raw examples per batch.")
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--ast-cache-dir",
        default="train_outputs/ast_cache",
        help="Directory for cached AST graph tensors. Set to empty string to disable.",
    )
    parser.add_argument(
        "--state-contrastive-weight",
        type=float,
        default=0.2,
        help="Weight for the true-state vs corrupted-state margin loss. Set 0 to disable.",
    )
    parser.add_argument("--state-contrastive-margin", type=float, default=0.01)
    parser.add_argument(
        "--state-contrastive-mode",
        choices=["none", "zero_all", "shuffle", "both"],
        default="zero_all",
        help="Corrupted state used by the contrastive loss. shuffle falls back to zero_all for one point.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--model-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-bitsandbytes-lora", action="store_true")
    parser.add_argument("--init-scem-checkpoint", default=None)

    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument(
        "--pretrain-region-points-per-example",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--pretrain-random-points-per-example",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--pretrain-even-points-per-example",
        type=int,
        default=32,
    )

    parser.add_argument("--adapt-epochs", type=int, default=2)
    parser.add_argument("--adapt-lr", type=float, default=5e-5)
    parser.add_argument(
        "--adapt-region-points-per-example",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--adapt-random-points-per-example",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--adapt-even-points-per-example",
        type=int,
        default=0,
    )

    # Compatibility attributes used by shared helpers.
    parser.set_defaults(
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    return parser.parse_args()


def load_records(path: str, max_raw_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            iterable = data
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            iterable = data["data"]
        elif isinstance(data, dict):
            iterable = [data]
        else:
            raise ValueError("JSON training file must contain an object or a list of objects")
        for record in iterable:
            if max_raw_examples is not None and len(records) >= max_raw_examples:
                break
            records.append(record)
        return records

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                if max_raw_examples is not None and len(records) >= max_raw_examples:
                    break
                records.append(json.loads(line))
    return records


class MultiPointRawDataset(Dataset):
    """Build many target points per raw record and run one backbone pass per record."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer,
        max_length: int,
        min_prefix_length: int,
        region_points_per_example: int,
        random_points_per_example: int,
        even_points_per_example: int,
        skip_overlength: bool,
        max_training_points: Optional[int],
        split_name: str,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_prefix_length = min_prefix_length
        self.region_points_per_example = region_points_per_example
        self.random_points_per_example = random_points_per_example
        self.even_points_per_example = even_points_per_example
        self.examples: List[MultiPointExample] = []

        raw_examples = 0
        skipped_overlength = 0
        point_count = 0
        for record in records:
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
            if len(ids) <= min_prefix_length:
                continue
            example = self._build_example(
                text=text,
                ids=ids,
                offsets=encoded.offset_mapping,
                target_char_start=target_char_start,
                remaining_points=None if max_training_points is None else max_training_points - point_count,
            )
            if example is None:
                continue
            self.examples.append(example)
            point_count += len(example.labels)
            if max_training_points is not None and point_count >= max_training_points:
                break

        if not self.examples:
            raise ValueError(f"No usable {split_name} examples found")
        message = (
            f"Loaded {point_count} {split_name} points from {len(self.examples)} usable raw examples "
            f"({raw_examples} records scanned)"
        )
        if skipped_overlength:
            message += f"; skipped {skipped_overlength} overlength records"
        print(message)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

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
        raise ValueError("Each record must contain text, prompt/completion, messages, or instruction/output")

    def _build_example(
        self,
        text: str,
        ids: List[int],
        offsets: Sequence[Tuple[int, int]],
        target_char_start: int,
        remaining_points: Optional[int],
    ) -> Optional[MultiPointExample]:
        min_target_pos = self._char_to_token_pos(target_char_start, offsets) or self.min_prefix_length
        min_target_pos = max(self.min_prefix_length, min_target_pos)
        if min_target_pos >= len(ids):
            return None

        positions: List[Tuple[int, str]] = []
        seen = set()
        self._add_region_positions(text, offsets, target_char_start, min_target_pos, ids, positions, seen)
        self._add_even_positions(min_target_pos, len(ids), positions, seen)
        self._add_random_positions(min_target_pos, len(ids), positions, seen)

        if not positions:
            positions.append((random.randint(min_target_pos, len(ids) - 1), "fallback"))
        if remaining_points is not None:
            positions = positions[: max(0, remaining_points)]
        if not positions:
            return None

        target_positions: List[int] = []
        labels: List[int] = []
        state_prefix_texts: List[str] = []
        for token_pos, _source in positions:
            prefix_end = offsets[token_pos][0] if token_pos < len(offsets) else len(text)
            prefix_end = max(target_char_start, prefix_end)
            target_positions.append(token_pos)
            labels.append(ids[token_pos])
            state_prefix_texts.append(
                text[:target_char_start] + GENERATED_PREFIX_MARKER + text[target_char_start:prefix_end]
            )
        return MultiPointExample(
            ids=ids,
            target_positions=target_positions,
            labels=labels,
            state_prefix_texts=state_prefix_texts,
        )

    def _add_region_positions(
        self,
        text: str,
        offsets: Sequence[Tuple[int, int]],
        target_char_start: int,
        min_target_pos: int,
        ids: List[int],
        positions: List[Tuple[int, str]],
        seen: set,
    ) -> None:
        budget = self.region_points_per_example
        if budget <= 0:
            return
        for region, anchors in REGION_ANCHORS.items():
            for anchor in anchors:
                search_start = target_char_start
                while len(positions) < budget:
                    char_pos = text.find(anchor, search_start)
                    if char_pos < 0:
                        break
                    token_pos = self._char_to_token_pos(min(len(text) - 1, char_pos + len(anchor)), offsets)
                    if self._is_valid_target(token_pos, len(ids), min_target_pos) and token_pos not in seen:
                        positions.append((token_pos, region))
                        seen.add(token_pos)
                    search_start = char_pos + len(anchor)
                if len(positions) >= budget:
                    return

    def _add_even_positions(
        self,
        min_target_pos: int,
        end_pos: int,
        positions: List[Tuple[int, str]],
        seen: set,
    ) -> None:
        budget = self.even_points_per_example
        if budget <= 0:
            return
        span = end_pos - min_target_pos
        if span <= 0:
            return
        for index in range(1, budget + 1):
            token_pos = min_target_pos + int(index * span / (budget + 1))
            token_pos = min(max(token_pos, min_target_pos), end_pos - 1)
            if token_pos not in seen:
                positions.append((token_pos, "even"))
                seen.add(token_pos)

    def _add_random_positions(
        self,
        min_target_pos: int,
        end_pos: int,
        positions: List[Tuple[int, str]],
        seen: set,
    ) -> None:
        budget = self.random_points_per_example
        if budget <= 0:
            return
        candidates = [pos for pos in range(min_target_pos, end_pos) if pos not in seen]
        for token_pos in random.sample(candidates, k=min(budget, len(candidates))):
            positions.append((token_pos, "random"))
            seen.add(token_pos)

    @staticmethod
    def _char_to_token_pos(char_pos: int, offsets: Sequence[Tuple[int, int]]) -> Optional[int]:
        for index, (start, end) in enumerate(offsets):
            if start == end:
                continue
            if start <= char_pos < end or char_pos < end:
                return index
        return None

    @staticmethod
    def _is_valid_target(token_pos: Optional[int], num_ids: int, min_target_pos: int) -> bool:
        return token_pos is not None and min_target_pos <= token_pos < num_ids


class MultiPointCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, examples: List[MultiPointExample]) -> MultiPointBatch:
        max_len = max(len(example.ids) for example in examples)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
        point_batch_indices: List[int] = []
        point_positions: List[int] = []
        labels: List[int] = []
        state_prefix_texts: List[str] = []
        for row, example in enumerate(examples):
            length = len(example.ids)
            input_ids[row, :length] = torch.tensor(example.ids, dtype=torch.long)
            attention_mask[row, :length] = 1
            for token_pos, label, state_prefix in zip(
                example.target_positions,
                example.labels,
                example.state_prefix_texts,
            ):
                point_batch_indices.append(row)
                point_positions.append(max(0, token_pos - 1))
                labels.append(label)
                state_prefix_texts.append(state_prefix)
        return MultiPointBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_batch_indices=torch.tensor(point_batch_indices, dtype=torch.long),
            point_positions=torch.tensor(point_positions, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            state_prefix_texts=state_prefix_texts,
        )


def load_scem_weights(path: str, scem: SCEModule) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    missing, unexpected = scem.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"SCEM checkpoint is not compatible. missing={missing}, unexpected={unexpected}")


def compute_batch_loss(model, scem, batch: MultiPointBatch, extractor, args, accelerator: Accelerator):
    input_ids = batch.input_ids.to(accelerator.device)
    attention_mask = batch.attention_mask.to(accelerator.device)
    point_batch_indices = batch.point_batch_indices.to(accelerator.device)
    point_positions = batch.point_positions.to(accelerator.device)
    labels = batch.labels.to(accelerator.device)
    state_batch = extractor.extract_batch(batch.state_prefix_texts, device=accelerator.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    last_hidden = outputs.hidden_states[-1][point_batch_indices, point_positions, :]
    last_logits = outputs.logits[point_batch_indices, point_positions, :]
    return compute_state_conditioned_loss(scem, last_hidden, last_logits, labels, state_batch, args), labels.numel()


@torch.no_grad()
def evaluate_validation(model, scem, dataloader, extractor, args, accelerator: Accelerator) -> float:
    model_was_training = model.training
    scem_was_training = scem.training
    model.eval()
    scem.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_count = torch.tensor(0.0, device=accelerator.device)
    for batch in dataloader:
        with accelerator.autocast():
            loss, count_value = compute_batch_loss(model, scem, batch, extractor, args, accelerator)
        count = torch.tensor(float(count_value), device=accelerator.device)
        total_loss += loss.detach().float() * count
        total_count += count
    stats = torch.stack([total_loss, total_count])
    gathered = accelerator.gather(stats).view(-1, 2)
    val_loss = gathered[:, 0].sum().item() / max(gathered[:, 1].sum().item(), 1.0)
    if model_was_training:
        model.train()
    if scem_was_training:
        scem.train()
    return val_loss


def build_dataset(args, tokenizer, records, stage: str, split_name: str, max_training_points=None):
    if stage == "pretrain":
        region_points = args.pretrain_region_points_per_example
        random_points = args.pretrain_random_points_per_example
        even_points = args.pretrain_even_points_per_example
    elif stage == "adapt":
        region_points = args.adapt_region_points_per_example
        random_points = args.adapt_random_points_per_example
        even_points = args.adapt_even_points_per_example
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return MultiPointRawDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=args.max_length,
        min_prefix_length=args.min_prefix_length,
        region_points_per_example=region_points,
        random_points_per_example=random_points,
        even_points_per_example=even_points,
        skip_overlength=args.skip_overlength,
        max_training_points=max_training_points,
        split_name=f"{stage} {split_name}",
    )


def train_stage(
    stage_name: str,
    model,
    scem,
    optimizer,
    scheduler,
    dataloader,
    val_dataloader,
    extractor,
    tokenizer,
    args,
    accelerator: Accelerator,
    logger: Optional[TrainingRunLogger],
    start_step: int,
) -> Tuple[int, float, int, Optional[float]]:
    global_step = start_step
    best_val_loss = float("inf")
    best_step = 0
    final_val_loss = None
    last_train_loss = None
    optimizer.zero_grad(set_to_none=True)

    epochs = args.pretrain_epochs if stage_name == "pretrain" else args.adapt_epochs
    for epoch in range(epochs):
        for batch in dataloader:
            with accelerator.accumulate(scem):
                with accelerator.autocast():
                    loss, _count = compute_batch_loss(model, scem, batch, extractor, args, accelerator)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(scem.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if global_step % args.log_steps == 0:
                current_loss = accelerator.gather(loss.detach()).mean().item()
                last_train_loss = current_loss
                lr = scheduler.get_last_lr()[0]
                accelerator.print(
                    f"{stage_name} epoch={epoch} step={global_step} loss={current_loss:.4f} lr={lr:.2e}"
                )
                if logger is not None:
                    logger.append(
                        event=f"{stage_name}_train",
                        epoch=epoch,
                        step=global_step,
                        train_loss=current_loss,
                        lr=lr,
                        best_val_loss=best_val_loss if math.isfinite(best_val_loss) else None,
                        best_step=best_step or None,
                    )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                metrics = {"stage": stage_name, "step": global_step}
                if last_train_loss is None:
                    last_train_loss = accelerator.gather(loss.detach()).mean().item()
                if val_dataloader is not None:
                    val_loss = evaluate_validation(model, scem, val_dataloader, extractor, args, accelerator)
                    metrics["val_loss"] = val_loss
                    accelerator.print(f"{stage_name} epoch={epoch} step={global_step} val_loss={val_loss:.4f}")
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
                                checkpoint_name=f"{stage_name}-best",
                                metrics={**metrics, "best_step": best_step},
                            )
                            if stage_name == "adapt":
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
                    if logger is not None:
                        logger.append(
                            event=f"{stage_name}_validation",
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
                        checkpoint_name=f"{stage_name}-step-{global_step}",
                        metrics=metrics,
                    )
                accelerator.wait_for_everyone()

    final_metrics = {"stage": stage_name, "step": global_step}
    if val_dataloader is not None:
        final_val_loss = evaluate_validation(model, scem, val_dataloader, extractor, args, accelerator)
        final_metrics["val_loss"] = final_val_loss
        accelerator.print(f"{stage_name} final step={global_step} val_loss={final_val_loss:.4f}")
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
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
                    checkpoint_name=f"{stage_name}-best",
                    metrics={**final_metrics, "best_step": best_step},
                )
                if stage_name == "adapt":
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
        if logger is not None:
            logger.append(
                event=f"{stage_name}_final",
                epoch=epochs,
                step=global_step,
                train_loss=last_train_loss,
                val_loss=final_val_loss,
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
            checkpoint_name="pretrain" if stage_name == "pretrain" else "final",
            metrics=final_metrics,
        )
    accelerator.wait_for_everyone()
    return global_step, best_val_loss, best_step, final_val_loss


def make_optimizer_and_scheduler(args, scem, dataloader, stage: str, accelerator: Accelerator):
    lr = args.pretrain_lr if stage == "pretrain" else args.adapt_lr
    epochs = args.pretrain_epochs if stage == "pretrain" else args.adapt_epochs
    optimizer = torch.optim.AdamW(scem.parameters(), lr=lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(dataloader) * epochs / (args.grad_accum_steps * accelerator.num_processes))
    total_steps = max(1, total_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler, total_steps


def log_initial_validation(
    model,
    scem,
    extractor,
    args,
    accelerator: Accelerator,
    logger: Optional[TrainingRunLogger],
    pretrain_val_loader,
    pretrain_scheduler,
    adapt_val_loader,
    adapt_scheduler,
) -> None:
    validation_runs = [
        ("pretrain", pretrain_val_loader, pretrain_scheduler),
        ("adapt", adapt_val_loader, adapt_scheduler),
    ]
    for stage_name, val_loader, scheduler in validation_runs:
        if val_loader is None:
            continue
        val_loss = evaluate_validation(model, scem, val_loader, extractor, args, accelerator)
        lr = scheduler.get_last_lr()[0] if scheduler is not None else None
        accelerator.print(f"{stage_name} initial step=0 val_loss={val_loss:.4f}")
        if logger is not None:
            logger.append(
                event=f"{stage_name}_initial_validation",
                epoch=None,
                step=0,
                val_loss=val_loss,
                lr=lr,
                update_plot=True,
            )


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
    model.eval()

    scem_config = build_scem_config(args, get_lm_config(model))
    scem = SCEModule(scem_config).to(dtype=next(model.parameters()).dtype)
    if args.init_scem_checkpoint:
        load_scem_weights(args.init_scem_checkpoint, scem)
    scem.train()

    raw_records = load_records(args.train_file, max_raw_examples=args.max_raw_examples)
    train_records, val_records = split_train_validation_records(raw_records, args.val_ratio, args.seed)
    if accelerator.is_main_process and val_records:
        print(f"Validation split: {len(val_records)} raw examples; training split: {len(train_records)} raw examples")

    collator = MultiPointCollator(tokenizer)
    pretrain_train = build_dataset(args, tokenizer, train_records, "pretrain", "training", args.max_training_points)
    pretrain_val = build_dataset(args, tokenizer, val_records, "pretrain", "validation") if val_records else None
    adapt_train = build_dataset(args, tokenizer, train_records, "adapt", "training", args.max_training_points)
    adapt_val = build_dataset(args, tokenizer, val_records, "adapt", "validation") if val_records else None

    pretrain_loader = DataLoader(pretrain_train, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    pretrain_val_loader = (
        DataLoader(pretrain_val, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
        if pretrain_val is not None
        else None
    )
    adapt_loader = DataLoader(adapt_train, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    adapt_val_loader = (
        DataLoader(adapt_val, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
        if adapt_val is not None
        else None
    )

    pretrain_optimizer, pretrain_scheduler, pretrain_steps = make_optimizer_and_scheduler(args, scem, pretrain_loader, "pretrain", accelerator)
    adapt_optimizer, adapt_scheduler, adapt_steps = make_optimizer_and_scheduler(args, scem, adapt_loader, "adapt", accelerator)

    train_logger = None
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        train_logger = TrainingRunLogger(
            args=args,
            train_points=sum(len(example.labels) for example in pretrain_train.examples) + sum(len(example.labels) for example in adapt_train.examples),
            val_points=(
                (sum(len(example.labels) for example in pretrain_val.examples) if pretrain_val is not None else 0)
                + (sum(len(example.labels) for example in adapt_val.examples) if adapt_val is not None else 0)
            ),
            total_steps=pretrain_steps + adapt_steps,
        )
        print(f"Training metrics will be written to {train_logger.path}")
    accelerator.wait_for_everyone()

    prepared = accelerator.prepare(
        model,
        scem,
        pretrain_optimizer,
        pretrain_scheduler,
        pretrain_loader,
        adapt_optimizer,
        adapt_scheduler,
        adapt_loader,
        *(tuple([pretrain_val_loader]) if pretrain_val_loader is not None else tuple()),
        *(tuple([adapt_val_loader]) if adapt_val_loader is not None else tuple()),
    )
    model = prepared[0]
    scem = prepared[1]
    pretrain_optimizer = prepared[2]
    pretrain_scheduler = prepared[3]
    pretrain_loader = prepared[4]
    adapt_optimizer = prepared[5]
    adapt_scheduler = prepared[6]
    adapt_loader = prepared[7]
    offset = 8
    if pretrain_val_loader is not None:
        pretrain_val_loader = prepared[offset]
        offset += 1
    if adapt_val_loader is not None:
        adapt_val_loader = prepared[offset]

    extractor = build_ast_extractor(scem_config, args)
    log_initial_validation(
        model=model,
        scem=scem,
        extractor=extractor,
        args=args,
        accelerator=accelerator,
        logger=train_logger,
        pretrain_val_loader=pretrain_val_loader,
        pretrain_scheduler=pretrain_scheduler,
        adapt_val_loader=adapt_val_loader,
        adapt_scheduler=adapt_scheduler,
    )
    global_step, _best1, _best_step1, _final1 = train_stage(
        "pretrain",
        model,
        scem,
        pretrain_optimizer,
        pretrain_scheduler,
        pretrain_loader,
        pretrain_val_loader,
        extractor,
        tokenizer,
        args,
        accelerator,
        train_logger,
        start_step=0,
    )
    global_step, best2, best_step2, final2 = train_stage(
        "adapt",
        model,
        scem,
        adapt_optimizer,
        adapt_scheduler,
        adapt_loader,
        adapt_val_loader,
        extractor,
        tokenizer,
        args,
        accelerator,
        train_logger,
        start_step=global_step,
    )
    if train_logger is not None:
        train_logger.finish(global_step, best_step2, best2, final2)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
