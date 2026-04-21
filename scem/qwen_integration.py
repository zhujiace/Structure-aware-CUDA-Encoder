from __future__ import annotations

from types import MethodType
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .decoding import SCEMLogitsProcessor
from .model import SCEModule
from .states import CudaProgramStateBatch, CudaProgramStateExtractor


InputStateProvider = Callable[[torch.LongTensor], CudaProgramStateBatch]


class TokenizerCudaStateProvider:
    """Build CUDA prefix states from generated token IDs via a tokenizer."""

    def __init__(
        self,
        tokenizer,
        extractor: Optional[CudaProgramStateExtractor] = None,
        skip_special_tokens: bool = False,
    ):
        self.tokenizer = tokenizer
        self.extractor = extractor or CudaProgramStateExtractor()
        self.skip_special_tokens = skip_special_tokens

    def __call__(self, input_ids: torch.LongTensor) -> CudaProgramStateBatch:
        prefixes = self.tokenizer.batch_decode(
            input_ids.detach().cpu(),
            skip_special_tokens=self.skip_special_tokens,
        )
        states = self.extractor.extract_batch(prefixes)
        return CudaProgramStateBatch.from_states(states, device=input_ids.device)


class SCEMForCausalLM(nn.Module):
    """Thin wrapper that adds SCEM bias to a causal LM's forward logits.

    This is useful for training or direct ``model(...)`` calls. For
    ``generate()``, prefer ``attach_scem_hidden_state_capture`` plus
    ``build_scem_logits_processor`` because generation with KV cache may pass
    only the latest token into ``forward``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        scem: SCEModule,
        state_provider: InputStateProvider,
        alpha: float = 1.0,
        recompute_loss: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.scem = scem
        self.state_provider = state_provider
        self.alpha = alpha
        self.recompute_loss = recompute_loss

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None, **kwargs):
        kwargs["output_hidden_states"] = True
        outputs = self.backbone(input_ids=input_ids, labels=labels, **kwargs)
        return add_scem_to_outputs(
            outputs=outputs,
            input_ids=input_ids,
            labels=labels,
            scem=self.scem,
            state_provider=self.state_provider,
            alpha=self.alpha,
            recompute_loss=self.recompute_loss,
        )

    def generate(self, *args, **kwargs):
        raise RuntimeError(
            "SCEMForCausalLM is intended for direct forward/training. "
            "For generation, use attach_scem_hidden_state_capture() and "
            "build_scem_logits_processor()."
        )


def attach_scem_to_causal_lm(
    model,
    scem: SCEModule,
    state_provider: InputStateProvider,
    alpha: float = 1.0,
    recompute_loss: bool = True,
):
    """Patch a loaded HF causal LM instance so its logits include SCEM bias.

    Use this for direct forward/training. For HF ``generate()``, use
    ``attach_scem_hidden_state_capture`` and ``build_scem_logits_processor`` so
    state extraction sees the complete generated prefix.

    The original forward is stored on ``model._scem_original_forward``.
    """

    if hasattr(model, "_scem_original_forward"):
        raise ValueError("SCEM is already attached to this model instance")

    model._scem_original_forward = model.forward
    model.scem = scem
    model.scem_state_provider = state_provider
    model.scem_alpha = alpha
    model.scem_recompute_loss = recompute_loss

    def forward_with_scem(self, input_ids: Optional[torch.LongTensor] = None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("SCEM integration requires input_ids to extract prefix state")
        kwargs["output_hidden_states"] = True
        outputs = self._scem_original_forward(input_ids=input_ids, labels=labels, **kwargs)
        return add_scem_to_outputs(
            outputs=outputs,
            input_ids=input_ids,
            labels=labels,
            scem=self.scem,
            state_provider=self.scem_state_provider,
            alpha=self.scem_alpha,
            recompute_loss=self.scem_recompute_loss,
        )

    model.forward = MethodType(forward_with_scem, model)
    return model


def attach_scem_hidden_state_capture(model):
    """Patch a HF causal LM to store the current step's last hidden state."""

    if hasattr(model, "_scem_capture_original_forward"):
        return model

    model._scem_capture_original_forward = model.forward
    model._scem_last_hidden_state = None

    def forward_with_hidden_capture(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        outputs = self._scem_capture_original_forward(*args, **kwargs)
        self._scem_last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        return outputs

    model.forward = MethodType(forward_with_hidden_capture, model)
    return model


def detach_scem_hidden_state_capture(model):
    """Restore a model patched by ``attach_scem_hidden_state_capture``."""

    if not hasattr(model, "_scem_capture_original_forward"):
        return model
    model.forward = model._scem_capture_original_forward
    delattr(model, "_scem_capture_original_forward")
    if hasattr(model, "_scem_last_hidden_state"):
        delattr(model, "_scem_last_hidden_state")
    return model


def build_scem_logits_processor(
    model,
    scem: SCEModule,
    state_provider: InputStateProvider,
    alpha: float = 1.0,
) -> SCEMLogitsProcessor:
    """Create the generation-time logits processor for a captured Qwen model."""

    def hidden_state_provider() -> torch.Tensor:
        hidden_state = getattr(model, "_scem_last_hidden_state", None)
        if hidden_state is None:
            raise RuntimeError(
                "No hidden state captured yet. Call attach_scem_hidden_state_capture(model) "
                "before model.generate(...)."
            )
        return hidden_state

    return SCEMLogitsProcessor(
        scem=scem,
        state_provider=state_provider,
        hidden_state_provider=hidden_state_provider,
        alpha=alpha,
    )


def detach_scem_from_causal_lm(model):
    """Restore a model patched by ``attach_scem_to_causal_lm``."""

    if not hasattr(model, "_scem_original_forward"):
        return model
    model.forward = model._scem_original_forward
    delattr(model, "_scem_original_forward")
    for attr in ("scem", "scem_state_provider", "scem_alpha", "scem_recompute_loss"):
        if hasattr(model, attr):
            delattr(model, attr)
    return model


def add_scem_to_outputs(
    outputs,
    input_ids: torch.LongTensor,
    labels: Optional[torch.LongTensor],
    scem: SCEModule,
    state_provider: InputStateProvider,
    alpha: float,
    recompute_loss: bool,
):
    hidden_state = outputs.hidden_states[-1][:, -1, :]
    state = state_provider(input_ids).to(hidden_state.device)
    scem_output = scem(hidden_state=hidden_state, state=state)

    logits = outputs.logits
    adjusted_logits = logits.clone()
    adjusted_logits[:, -1, :] = adjusted_logits[:, -1, :] + alpha * scem_output.bias.to(
        dtype=adjusted_logits.dtype
    )
    outputs.logits = adjusted_logits

    if labels is not None and recompute_loss:
        outputs.loss = causal_lm_loss(adjusted_logits, labels)
    outputs.scem_bias = scem_output.bias
    outputs.scem_context = scem_output.context
    return outputs


def causal_lm_loss(logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
