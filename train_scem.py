import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

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


class PrefixNextTokenDataset(Dataset):
    """Sample one next-token training step from each CUDA example."""

    def __init__(self, path: str, tokenizer, max_length: int, min_prefix_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_prefix_length = min_prefix_length
        self.examples = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = self._record_to_text(record)
                ids = tokenizer(text, add_special_tokens=False).input_ids[:max_length]
                if len(ids) > min_prefix_length:
                    self.examples.append(ids)
        if not self.examples:
            raise ValueError("No usable training examples found")

    def _record_to_text(self, record: Dict[str, Any]) -> str:
        if "messages" in record:
            return self.tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        if "prompt" in record and "completion" in record:
            messages = [{"role": "user", "content": record["prompt"]}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt + record["completion"]
        if "text" in record:
            return record["text"]
        raise ValueError("Each JSONL row must contain text, prompt/completion, or messages")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ids = self.examples[index]
        target_pos = random.randint(self.min_prefix_length, len(ids) - 1)
        return {
            "prefix_ids": ids[:target_pos],
            "label": ids[target_pos],
        }


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
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda" and dtype == torch.float16)
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
