import argparse

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


QWEN35_GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 20,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with Qwen3.5-0.8B plus SCEM logits bias.")
    parser.add_argument("--model-path", default="./models/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="Write a CUDA kernel for vector addition.")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--alpha", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype)
    model.to(device)
    model.eval()

    scem_config = SCEMConfig.from_lm_config(model.config)
    scem = SCEModule(scem_config).to(device=device, dtype=next(model.parameters()).dtype)
    scem.eval()

    state_provider = TokenizerCudaStateProvider(
        tokenizer=tokenizer,
        extractor=CudaProgramStateExtractor(task_family="elementwise", tensor_rank=1),
    )

    attach_scem_hidden_state_capture(model)
    scem_processor = build_scem_logits_processor(
        model=model,
        scem=scem,
        state_provider=state_provider,
        alpha=args.alpha,
    )

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            logits_processor=LogitsProcessorList([scem_processor]),
            **QWEN35_GENERATION_KWARGS,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
