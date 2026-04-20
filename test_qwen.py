from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./models/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path)
print(type(model))
print("Last decoder layer:")
print(model.model.layers[-1])

print("\nFinal norm before lm_head:")
print(model.model.norm)

print("\nLM head:")
print(model.lm_head)