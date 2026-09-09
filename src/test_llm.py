import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

print("CUDA available:", torch.cuda.is_available())
print("Model device:", model.device)

messages = [
    {
        "role": "user",
        "content": "Explain in one sentence why recycling plastic is useful."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)

# Make sure we have the actual tensor
if hasattr(inputs, "input_ids"):
    input_ids = inputs.input_ids
else:
    input_ids = inputs

input_ids = input_ids.to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=60,
        do_sample=False
    )

answer = tokenizer.decode(
    outputs[0][input_ids.shape[-1]:],
    skip_special_tokens=True
)

print("\nModel response:")
print(answer)