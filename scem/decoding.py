from __future__ import annotations

from typing import Callable

import torch
from transformers import LogitsProcessor

from .model import SCEModule
from .states import CudaASTGraphBatch


StateProvider = Callable[[torch.LongTensor], CudaASTGraphBatch]
HiddenStateProvider = Callable[[], torch.Tensor]


class SCEMLogitsProcessor(LogitsProcessor):
    """Hugging Face logits processor that adds SCEM bias during decoding.

    ``generate`` only passes ``input_ids`` and ``scores`` into a logits
    processor, so the current decoder hidden state must be supplied by a small
    wrapper/hook through ``hidden_state_provider``. This keeps SCEM separate from
    the backbone model weights.
    """

    def __init__(
        self,
        scem: SCEModule,
        state_provider: StateProvider,
        hidden_state_provider: HiddenStateProvider,
        alpha: float = 1.0,
    ):
        self.scem = scem
        self.state_provider = state_provider
        self.hidden_state_provider = hidden_state_provider
        self.alpha = alpha
        self._logged_once = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        hidden_state = self.hidden_state_provider()
        state = self.state_provider(input_ids).to(scores.device)
        output = self.scem(hidden_state=hidden_state.to(scores.device), state=state)
        if not self._logged_once:
            mean_abs_bias = output.bias.abs().mean().item()
            print(
                f"[SCEM] first logits adjustment applied: alpha={self.alpha} "
                f"mean_abs_bias={mean_abs_bias:.6f}"
            )
            self._logged_once = True
        return scores + self.alpha * output.bias.to(dtype=scores.dtype)


def make_static_state_provider(batch: CudaASTGraphBatch) -> StateProvider:
    """Create a state provider for tests or fixed-prefix decoding experiments."""

    def provider(input_ids: torch.LongTensor) -> CudaASTGraphBatch:
        if input_ids.shape[0] != batch.batch_size:
            raise ValueError("input_ids batch size does not match the fixed SCEM state batch")
        return batch

    return provider
