import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scem import CudaASTGraphBatch, CudaASTGraphExtractor, SCEMConfig, SCEModule
from train import PrefixCollator, PrefixNextTokenDataset, get_lm_config, split_train_validation_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect SCEM bias strength and its effect on backbone next-token logits."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True, help="Training-format JSON/JSONL file used to build prefix points.")
    parser.add_argument(
        "--scem-checkpoint",
        action="append",
        default=[],
        help="SCEM checkpoint path, or LABEL=PATH. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for summary.json and summary.csv.")
    parser.add_argument("--run-name", default=None, help="Optional metadata label stored in output JSON.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-raw-examples", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-prefix-length", type=int, default=16)
    parser.add_argument("--skip-overlength", action="store_true")
    parser.add_argument("--region-points-per-example", type=int, default=8)
    parser.add_argument("--random-points-per-example", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--model-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=9.0,
        help="Count |bias| above this threshold. Default matches 90%% of SCEM max_bias=10.",
    )
    parser.add_argument(
        "--include-baseline-row",
        action="store_true",
        help="Include a backbone-only row with CE/rank metrics and empty bias metrics.",
    )
    parser.add_argument(
        "--state-ablation",
        action="append",
        choices=["true", "zero_all", "shuffled"],
        default=None,
        help="State variant to evaluate. Can be passed multiple times. Default: true.",
    )
    parser.add_argument(
        "--shuffle-offset",
        type=int,
        default=None,
        help="Offset used by --state-ablation shuffled. Default: half the inspected point count.",
    )
    return parser.parse_args()


def resolve_model_dtype(value: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def parse_checkpoint_spec(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        label = label.strip()
        checkpoint = Path(path.strip())
    else:
        checkpoint = Path(spec)
        if checkpoint.name == "scem.pt" and checkpoint.parent.name:
            label = checkpoint.parent.name
        else:
            label = checkpoint.stem
    if not label:
        raise ValueError(f"Empty checkpoint label in {spec!r}")
    return label, checkpoint


def config_from_checkpoint(checkpoint: Dict[str, Any], model) -> SCEMConfig:
    config = checkpoint.get("config")
    if isinstance(config, SCEMConfig):
        return config
    if isinstance(config, dict):
        return SCEMConfig(**config)
    return SCEMConfig.from_lm_config(get_lm_config(model))


def load_scem(label: str, path: Path, model, dtype: torch.dtype, device: torch.device) -> SCEModule:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError(f"{path} does not contain a SCEM state_dict")
    config = config_from_checkpoint(checkpoint, model)
    state_dict = checkpoint["state_dict"]
    scem = SCEModule(config).to(device=device, dtype=dtype)
    missing, unexpected = scem.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"{label} checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    scem.eval()
    return scem


def select_records(args) -> Tuple[List[Dict[str, Any]], int, int]:
    records = PrefixNextTokenDataset.load_records(args.train_file, max_raw_examples=args.max_raw_examples)
    if args.split == "all":
        return records, len(records), 0
    train_records, val_records = split_train_validation_records(records, args.val_ratio, args.seed)
    if args.split == "train":
        return train_records, len(train_records), len(val_records)
    if not val_records:
        raise ValueError("--split val requires --val-ratio > 0 and at least two records")
    return val_records, len(train_records), len(val_records)


def build_dataloader(args, tokenizer):
    records, train_count, val_count = select_records(args)
    dataset = PrefixNextTokenDataset(
        path=None,
        tokenizer=tokenizer,
        max_length=args.max_length,
        min_prefix_length=args.min_prefix_length,
        region_points_per_example=args.region_points_per_example,
        random_points_per_example=args.random_points_per_example,
        skip_overlength=args.skip_overlength,
        max_training_points=args.max_points,
        records=records,
        split_name=f"inspect-{args.split}",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PrefixCollator(tokenizer),
        num_workers=0,
    )
    return dataloader, dataset, len(dataset), train_count, val_count


class RunningStats:
    def __init__(
        self,
        label: str,
        checkpoint: Optional[str],
        checkpoint_label: Optional[str] = None,
        state_ablation: Optional[str] = None,
    ):
        self.label = label
        self.checkpoint = checkpoint
        self.checkpoint_label = checkpoint_label
        self.state_ablation = state_ablation
        self.count = 0
        self.sums: Dict[str, float] = {}
        self.max_values: Dict[str, float] = {}

    def add(self, name: str, values: torch.Tensor) -> None:
        values = values.detach().float()
        if values.numel() == 0:
            return
        self.sums[name] = self.sums.get(name, 0.0) + values.sum().item()

    def add_scalar_sum(self, name: str, value: float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value)

    def update_max(self, name: str, values: torch.Tensor) -> None:
        values = values.detach().float()
        if values.numel() == 0:
            return
        current = values.max().item()
        self.max_values[name] = max(self.max_values.get(name, -float("inf")), current)

    def add_count(self, count: int) -> None:
        self.count += int(count)

    def mean(self, name: str) -> Optional[float]:
        if self.count == 0 or name not in self.sums:
            return None
        return self.sums[name] / self.count

    def finalize(self, args) -> Dict[str, Any]:
        row = {
            "label": self.label,
            "checkpoint": self.checkpoint,
            "checkpoint_label": self.checkpoint_label,
            "state_ablation": self.state_ablation,
            "points": self.count,
            "alpha": args.alpha,
            "top_k": args.top_k,
            "saturation_threshold": args.saturation_threshold,
        }
        mean_fields = [
            "ce_base",
            "ce_adjusted",
            "ce_delta",
            "ce_delta_positive",
            "mean_abs_bias",
            "rms_bias",
            "p95_abs_bias",
            "saturation_frac",
            "gold_bias",
            "gold_bias_abs",
            "gold_rank_base",
            "gold_rank_adjusted",
            "gold_rank_delta",
            "top1_changed",
            "topk_overlap",
            "kl_adjusted_vs_base",
        ]
        for field in mean_fields:
            row[field] = self.mean(field)
        row["max_abs_bias"] = self.max_values.get("max_abs_bias")
        if row["ce_base"] is not None:
            row["ppl_base"] = math.exp(min(50.0, row["ce_base"]))
        else:
            row["ppl_base"] = None
        if row["ce_adjusted"] is not None:
            row["ppl_adjusted"] = math.exp(min(50.0, row["ce_adjusted"]))
        else:
            row["ppl_adjusted"] = None
        return row


def rank_of_label(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    gold = logits.gather(1, labels[:, None])
    return (logits > gold).sum(dim=1).float() + 1.0


def topk_overlap(base_logits: torch.Tensor, adjusted_logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, base_logits.size(-1))
    base_top = torch.topk(base_logits, k=k, dim=-1).indices
    adjusted_top = torch.topk(adjusted_logits, k=k, dim=-1).indices
    matches = base_top[:, :, None].eq(adjusted_top[:, None, :]).any(dim=2).float().sum(dim=1)
    return matches / float(k)


def update_baseline_stats(stats: RunningStats, logits: torch.Tensor, labels: torch.Tensor) -> None:
    ce_base = F.cross_entropy(logits, labels, reduction="none")
    gold_rank = rank_of_label(logits, labels)
    count = labels.numel()
    stats.add_count(count)
    stats.add("ce_base", ce_base)
    stats.add("ce_adjusted", ce_base)
    stats.add_scalar_sum("ce_delta", 0.0)
    stats.add_scalar_sum("ce_delta_positive", 0.0)
    stats.add("gold_rank_base", gold_rank)
    stats.add("gold_rank_adjusted", gold_rank)
    stats.add_scalar_sum("gold_rank_delta", 0.0)
    stats.add_scalar_sum("top1_changed", 0.0)
    stats.add_scalar_sum("topk_overlap", float(count))
    stats.add_scalar_sum("kl_adjusted_vs_base", 0.0)


def update_scem_stats(
    stats: RunningStats,
    scem: SCEModule,
    hidden: torch.Tensor,
    states: CudaASTGraphBatch,
    logits: torch.Tensor,
    labels: torch.Tensor,
    args,
) -> None:
    output = scem(hidden, states)
    bias = output.bias.float()
    adjusted = logits + args.alpha * bias

    ce_base = F.cross_entropy(logits, labels, reduction="none")
    ce_adjusted = F.cross_entropy(adjusted, labels, reduction="none")
    ce_delta = ce_base - ce_adjusted
    abs_bias = bias.abs()
    mean_abs = abs_bias.mean(dim=1)
    rms = bias.square().mean(dim=1).sqrt()
    p95 = torch.quantile(abs_bias, q=0.95, dim=1)
    max_abs = abs_bias.max(dim=1).values
    saturation = (abs_bias > args.saturation_threshold).float().mean(dim=1)
    gold_bias = bias.gather(1, labels[:, None]).squeeze(1)

    base_rank = rank_of_label(logits, labels)
    adjusted_rank = rank_of_label(adjusted, labels)
    base_top1 = logits.argmax(dim=1)
    adjusted_top1 = adjusted.argmax(dim=1)
    top1_changed = base_top1.ne(adjusted_top1).float()
    overlap = topk_overlap(logits, adjusted, args.top_k)

    logp_adjusted = F.log_softmax(adjusted, dim=1)
    logp_base = F.log_softmax(logits, dim=1)
    kl_adjusted_vs_base = (logp_adjusted.exp() * (logp_adjusted - logp_base)).sum(dim=1)

    count = labels.numel()
    stats.add_count(count)
    stats.add("ce_base", ce_base)
    stats.add("ce_adjusted", ce_adjusted)
    stats.add("ce_delta", ce_delta)
    stats.add("ce_delta_positive", ce_delta.gt(0).float())
    stats.add("mean_abs_bias", mean_abs)
    stats.add("rms_bias", rms)
    stats.add("p95_abs_bias", p95)
    stats.add("saturation_frac", saturation)
    stats.add("gold_bias", gold_bias)
    stats.add("gold_bias_abs", gold_bias.abs())
    stats.add("gold_rank_base", base_rank)
    stats.add("gold_rank_adjusted", adjusted_rank)
    stats.add("gold_rank_delta", base_rank - adjusted_rank)
    stats.add("top1_changed", top1_changed)
    stats.add("topk_overlap", overlap)
    stats.add("kl_adjusted_vs_base", kl_adjusted_vs_base)
    stats.update_max("max_abs_bias", max_abs)


def make_state_batch_for_ablation(
    mode: str,
    true_batch: CudaASTGraphBatch,
    extractor: CudaASTGraphExtractor,
    all_prefixes: Sequence[str],
    batch_start: int,
    batch_size: int,
    shuffle_offset: int,
    device: torch.device,
) -> CudaASTGraphBatch:
    if mode == "true":
        return true_batch
    if mode == "shuffled":
        if not all_prefixes:
            return true_batch
        state_count = len(all_prefixes)
        shuffled_prefixes = [all_prefixes[(batch_start + index + shuffle_offset) % state_count] for index in range(batch_size)]
        return extractor.extract_batch(shuffled_prefixes, device=device)
    if mode == "zero_all":
        return true_batch.zero_like()
    raise ValueError(f"Unsupported state ablation: {mode}")


def make_extractor_from_config(config: SCEMConfig) -> CudaASTGraphExtractor:
    return CudaASTGraphExtractor(
        max_nodes=config.ast_max_nodes,
        max_edges=config.ast_max_edges,
        node_type_vocab_size=config.ast_node_type_vocab_size,
        edge_type_vocab_size=config.ast_edge_type_vocab_size,
        text_vocab_size=config.ast_text_vocab_size,
        max_depth=config.ast_max_depth,
        max_child_index=config.ast_max_child_index,
        node_flag_dim=config.ast_node_flag_dim,
        node_position_dim=config.ast_node_position_dim,
    )


def rows_to_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def config_to_dict(config: SCEMConfig) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    return dict(config.__dict__)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_model_dtype(args.model_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    state_ablation_modes = args.state_ablation or ["true"]
    dataloader, dataset, point_count, train_count, val_count = build_dataloader(args, tokenizer)
    all_prefixes = [example.state_prefix for example in dataset.examples]
    shuffle_offset = args.shuffle_offset
    if shuffle_offset is None:
        shuffle_offset = max(1, point_count // 2)

    checkpoint_specs = [parse_checkpoint_spec(spec) for spec in args.scem_checkpoint]
    scems: List[Tuple[str, str, SCEModule]] = []
    for label, checkpoint_path in checkpoint_specs:
        scems.append((label, str(checkpoint_path), load_scem(label, checkpoint_path, model, dtype, device)))
    extractor = make_extractor_from_config(scems[0][2].config) if scems else CudaASTGraphExtractor()

    stats: List[RunningStats] = []
    if args.include_baseline_row:
        stats.append(RunningStats("baseline", None, checkpoint_label="baseline", state_ablation=None))
    scem_stats: Dict[Tuple[str, str], RunningStats] = {}
    for label, checkpoint_path, _scem in scems:
        for ablation in state_ablation_modes:
            stat = RunningStats(
                label=f"{label}:{ablation}",
                checkpoint=checkpoint_path,
                checkpoint_label=label,
                state_ablation=ablation,
            )
            scem_stats[(label, ablation)] = stat
            stats.append(stat)

    with torch.no_grad():
        processed_points = 0
        for batch_index, batch in enumerate(dataloader, start=1):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            state_batch = extractor.extract_batch(batch.state_prefix_texts, device=device)
            batch_size = labels.numel()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1][:, -1, :]
            logits = outputs.logits[:, -1, :].float()

            if args.include_baseline_row:
                update_baseline_stats(stats[0], logits, labels)
            state_batches = {
                ablation: make_state_batch_for_ablation(
                    ablation,
                    state_batch,
                    extractor,
                    all_prefixes,
                    processed_points,
                    batch_size,
                    shuffle_offset,
                    device,
                )
                for ablation in state_ablation_modes
            }
            for label, _checkpoint_path, scem in scems:
                for ablation in state_ablation_modes:
                    update_scem_stats(
                        scem_stats[(label, ablation)],
                        scem,
                        hidden,
                        state_batches[ablation],
                        logits,
                        labels,
                        args,
                    )

            processed_points += batch_size

            if batch_index % 10 == 0:
                done = min(processed_points, point_count)
                print(f"processed {done}/{point_count} points", flush=True)

    rows = [stat.finalize(args) for stat in stats]
    metadata = {
        "run_name": args.run_name,
        "model_path": args.model_path,
        "train_file": args.train_file,
        "split": args.split,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "train_raw_records": train_count,
        "val_raw_records": val_count,
        "points": point_count,
        "max_points": args.max_points,
        "max_length": args.max_length,
        "skip_overlength": args.skip_overlength,
        "region_points_per_example": args.region_points_per_example,
        "random_points_per_example": args.random_points_per_example,
        "batch_size": args.batch_size,
        "state_ablation": state_ablation_modes,
        "shuffle_offset": shuffle_offset,
        "model_dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "scem_config": config_to_dict(scems[0][2].config) if scems else None,
    }
    result = {"metadata": metadata, "rows": rows}

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        rows_to_csv(rows, output_dir / "summary.csv")
        print(f"Wrote {output_dir / 'summary.json'}")
        print(f"Wrote {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
