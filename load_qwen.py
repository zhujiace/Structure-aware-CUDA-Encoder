import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./models/Qwen3.5-0.8B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

prompt = "Do you have thinking mode?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))