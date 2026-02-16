import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model


def load_config():
    with open("/home/aryu_user/Arun/aiproject_staging/hrms_ai/hrms_ai_training/config.yaml", "r") as f:
        return yaml.safe_load(f)


def format_chat(example):
    text = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        text += f"<|{role}|>\n{content}\n"
    text += "<|assistant|>\n"
    return {"text": text}


def main():
    config = load_config()
    model_name = config["model_name"]

    # -------------------------
    # 4-bit Quantization (REQUIRED for 4GB GPU)
    # -------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # IMPORTANT: Use quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # -------------------------
    # LoRA (Safe settings)
    # -------------------------
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=config["lora"]["target_modules"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # -------------------------
    # Dataset
    # -------------------------
    dataset = load_dataset(
        "json",
        data_files={
            "train": config["dataset"]["train_file"],
        }
    )

    dataset = dataset.map(format_chat)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256,  # FORCE 256 for 4GB GPU
        )

    dataset = dataset.map(tokenize, batched=True)

    # -------------------------
    # Training Args (SAFE)
    # -------------------------
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="no",   # VERY IMPORTANT
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    model.save_pretrained(config["training"]["output_dir"])
    tokenizer.save_pretrained(config["training"]["output_dir"])


if __name__ == "__main__":
    main()
