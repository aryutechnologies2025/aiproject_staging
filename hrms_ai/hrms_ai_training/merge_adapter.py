from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "outputs/hrms_manager_adapter"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained("outputs/hrms_manager_full")
