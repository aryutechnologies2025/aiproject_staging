import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "meta-llama/Llama-2-13b-chat-hf")


def run_inference(prompt: str = "Explain Django in simple words.") -> str:
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading model (this may take time)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print(f"Model loaded: {model}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


if __name__ == "__main__":
    response = run_inference()
    print("\n--- Response ---\n")
    print(response)
