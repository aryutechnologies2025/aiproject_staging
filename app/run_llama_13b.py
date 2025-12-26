from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = r"C:\Users\aryue\.cache\huggingface\hub\models--meta-llama--Llama-2-13b-chat-hf\snapshots\a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model (this will take time)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,   # if GPU
    low_cpu_mem_usage=True,
    device_map="auto"            # uses GPU if available, else CPU
)
print(f'model: {model}')
prompt = "Explain Django in simple words."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating...")
outputs = model.generate(
    **inputs,
    max_new_tokens=15,
    temperature=0.7
)

print("\n--- Response ---\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
