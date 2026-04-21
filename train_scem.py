import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from scem import CudaProgramStateBatch, CudaProgramStateExtractor, SCEMConfig, SCEModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train SCEM on next-token CUDA generation steps.")
    parser.add_argument("--model-path", default="./models/Qwen3.5-0.8B")
    parser.add_argument("--train-file", required=True, help="JSONL file with text, prompt/completion, or messages.")
    parser.add_argument("--output-dir", default="./checkpoints/scem")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-prefix-length", type=int, default=16)
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
    parser.add_argument("--task-family", default="elementwise")
    parser.add_argument("--tensor-rank", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
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


class PrefixNextTokenDataset(Dataset):
    """Expand each CUDA example into multiple region-aware next-token steps."""

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        min_prefix_length: int,
        region_points_per_example: int,
        random_points_per_example: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_prefix_length = min_prefix_length
        self.region_points_per_example = region_points_per_example
        self.random_points_per_example = random_points_per_example
        self.examples = []
        raw_examples = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text, target_char_start = self._record_to_text(record)
                raw_examples += 1
                encoded = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=max_length,
                )
                ids = encoded.input_ids
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
        if not self.examples:
            raise ValueError("No usable training examples found")
        print(f"Loaded {len(self.examples)} training points from {raw_examples} raw examples")

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
        if "text" in record:
            return record["text"], 0
        raise ValueError("Each JSONL row must contain text, prompt/completion, or messages")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        point = self.examples[index]
        return {
            "prefix_ids": point.ids[: point.target_pos],
            "label": point.ids[point.target_pos],
            "source": point.source,
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
                token_pos = self._char_to_token_pos(char_pos, offsets)
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

        return [TrainingPoint(ids=ids, target_pos=token_pos, source=source) for token_pos, source in positions]

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
    prefix_texts: List[str]


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
        prefix_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return PrefixBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            prefix_texts=prefix_texts,
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
        model.print_trainable_parameters()
    return model


def save_checkpoint(output_dir: str, scem: SCEModule, model, tokenizer, args, step: int):
    path = os.path.join(output_dir, f"step-{step}")
    os.makedirs(path, exist_ok=True)
    torch.save({"config": scem.config, "state_dict": scem.state_dict()}, os.path.join(path, "scem.pt"))
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, "training_args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    if args.use_lora and hasattr(model, "save_pretrained"):
        model.save_pretrained(os.path.join(path, "lora"))


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype)
    model = configure_backbone(model, args)
    model.to(device)
    model.train(args.use_lora)
    if not args.use_lora:
        model.eval()

    scem_config = SCEMConfig.from_lm_config(model.config, bias_rank=args.bias_rank)
    scem = SCEModule(scem_config).to(device=device, dtype=next(model.parameters()).dtype)
    scem.train()

    dataset = PrefixNextTokenDataset(
        path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        min_prefix_length=args.min_prefix_length,
        region_points_per_example=args.region_points_per_example,
        random_points_per_example=args.random_points_per_example,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PrefixCollator(tokenizer),
        num_workers=0,
    )

    trainable_params = list(scem.parameters()) + [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(dataloader) * args.epochs / args.grad_accum_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and dtype == torch.float16)
    extractor = CudaProgramStateExtractor(task_family=args.task_family, tensor_rank=args.tensor_rank)

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for local_step, batch in enumerate(dataloader):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            states = extractor.extract_batch(batch.prefix_texts)
            state_batch = CudaProgramStateBatch.from_states(states, device=device)

            autocast_enabled = device == "cuda"
            with torch.autocast(device_type=device, dtype=dtype, enabled=autocast_enabled):
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
                loss = F.cross_entropy(adjusted_logits, labels)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            should_step = (local_step + 1) % args.grad_accum_steps == 0
            should_step = should_step or (local_step + 1) == len(dataloader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_steps == 0:
                    current_loss = loss.item() * args.grad_accum_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"epoch={epoch} step={global_step} loss={current_loss:.4f} lr={lr:.2e}")

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(args.output_dir, scem, model, tokenizer, args, global_step)

    save_checkpoint(args.output_dir, scem, model, tokenizer, args, global_step)
    print(f"Training complete. Saved checkpoint to {args.output_dir}/step-{global_step}")


if __name__ == "__main__":
    main()
