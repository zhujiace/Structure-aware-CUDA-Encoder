from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


TASK_FAMILIES = {
    "unknown": 0,
    "elementwise": 1,
    "reduction": 2,
    "matmul": 3,
    "stencil": 4,
    "scan": 5,
    "transpose": 6,
    "convolution": 7,
}

PROGRAM_REGIONS = {
    "unknown": 0,
    "signature": 1,
    "setup": 2,
    "indexing": 3,
    "guard": 4,
    "shared_memory": 5,
    "compute": 6,
    "write_back": 7,
}


@dataclass(frozen=True)
class CudaProgramState:
    """Compact CUDA program state for one decoding prefix."""

    task_family: int = 0
    tensor_rank: int = 0
    program_region: int = 0
    may_need_guard: bool = False
    may_need_shared_memory: bool = False
    may_need_synchronization: bool = False
    has_index_definition: bool = False
    has_open_guard: bool = False
    has_shared_memory: bool = False
    has_syncthreads: bool = False
    has_write_back: bool = False
    braces_balanced: bool = True
    statement_stable: bool = True
    brace_depth: float = 0.0
    paren_depth: float = 0.0


@dataclass
class CudaProgramStateBatch:
    task_family: torch.LongTensor
    tensor_rank: torch.LongTensor
    program_region: torch.LongTensor
    static_flags: torch.Tensor
    prefix_flags: torch.Tensor
    numeric_features: torch.Tensor

    def to(self, device: torch.device | str) -> "CudaProgramStateBatch":
        return CudaProgramStateBatch(
            task_family=self.task_family.to(device),
            tensor_rank=self.tensor_rank.to(device),
            program_region=self.program_region.to(device),
            static_flags=self.static_flags.to(device),
            prefix_flags=self.prefix_flags.to(device),
            numeric_features=self.numeric_features.to(device),
        )

    @classmethod
    def from_states(
        cls,
        states: Sequence[CudaProgramState],
        device: torch.device | str | None = None,
    ) -> "CudaProgramStateBatch":
        task_family = torch.tensor([s.task_family for s in states], dtype=torch.long, device=device)
        tensor_rank = torch.tensor([s.tensor_rank for s in states], dtype=torch.long, device=device)
        program_region = torch.tensor([s.program_region for s in states], dtype=torch.long, device=device)
        static_flags = torch.tensor(
            [
                [s.may_need_guard, s.may_need_shared_memory, s.may_need_synchronization]
                for s in states
            ],
            dtype=torch.float32,
            device=device,
        )
        prefix_flags = torch.tensor(
            [
                [
                    s.has_index_definition,
                    s.has_open_guard,
                    s.has_shared_memory,
                    s.has_syncthreads,
                    s.has_write_back,
                    s.braces_balanced,
                    s.statement_stable,
                ]
                for s in states
            ],
            dtype=torch.float32,
            device=device,
        )
        numeric_features = torch.tensor(
            [[s.brace_depth, s.paren_depth] for s in states],
            dtype=torch.float32,
            device=device,
        )
        return cls(
            task_family=task_family,
            tensor_rank=tensor_rank,
            program_region=program_region,
            static_flags=static_flags,
            prefix_flags=prefix_flags,
            numeric_features=numeric_features,
        )


class CudaProgramStateExtractor:
    """Heuristic prefix-state extractor used before a learned/parser extractor exists."""

    def __init__(self, task_family: str | int = "unknown", tensor_rank: int = 0):
        if isinstance(task_family, str):
            task_family = TASK_FAMILIES.get(task_family, TASK_FAMILIES["unknown"])
        self.task_family = task_family
        self.tensor_rank = tensor_rank

    def extract_batch(self, prefixes: Iterable[str]) -> List[CudaProgramState]:
        return [self.extract(prefix) for prefix in prefixes]

    def extract(self, prefix: str) -> CudaProgramState:
        lowered = prefix.lower()
        brace_depth = max(0, prefix.count("{") - prefix.count("}"))
        paren_depth = max(0, prefix.count("(") - prefix.count(")"))
        has_guard_token = any(token in prefix for token in ("if (", "if(", "&&", "||"))
        has_write_back = "=" in prefix and any(op in prefix for op in ("] =", "]= ", "]="))
        has_shared = "__shared__" in prefix or "extern __shared__" in prefix
        has_sync = "__syncthreads" in prefix
        has_index = any(
            token in prefix
            for token in (
                "threadIdx.",
                "blockIdx.",
                "blockDim.",
                "int idx",
                "int i",
                "int row",
                "int col",
            )
        )

        region = self._infer_region(prefix, has_guard_token, has_shared, has_write_back)
        return CudaProgramState(
            task_family=self.task_family,
            tensor_rank=self.tensor_rank,
            program_region=region,
            may_need_guard=self.tensor_rank > 0,
            may_need_shared_memory=any(token in lowered for token in ("matmul", "tile", "shared")),
            may_need_synchronization=has_shared or "shared" in lowered or "tile" in lowered,
            has_index_definition=has_index,
            has_open_guard=has_guard_token and prefix.rfind("if") > prefix.rfind("}"),
            has_shared_memory=has_shared,
            has_syncthreads=has_sync,
            has_write_back=has_write_back,
            braces_balanced=brace_depth == 0,
            statement_stable=prefix.rstrip().endswith((";", "{", "}")),
            brace_depth=float(min(brace_depth, 32)) / 32.0,
            paren_depth=float(min(paren_depth, 32)) / 32.0,
        )

    @staticmethod
    def _infer_region(prefix: str, has_guard: bool, has_shared: bool, has_write_back: bool) -> int:
        if "__global__" in prefix and "{" not in prefix:
            return PROGRAM_REGIONS["signature"]
        if has_write_back:
            return PROGRAM_REGIONS["write_back"]
        if has_guard:
            return PROGRAM_REGIONS["guard"]
        if has_shared:
            return PROGRAM_REGIONS["shared_memory"]
        if any(token in prefix for token in ("threadIdx.", "blockIdx.", "blockDim.")):
            return PROGRAM_REGIONS["indexing"]
        if "{" in prefix:
            return PROGRAM_REGIONS["compute"]
        return PROGRAM_REGIONS["setup"]
