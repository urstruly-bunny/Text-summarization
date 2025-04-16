# finetune.py

import logging
import torch
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
dataset = load_dataset("samsum")
train_data = dataset["train"]
val_data = dataset["validation"]

# Load tokenizer and model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Preprocess
def preprocess_function(batch):
    inputs = tokenizer(
        batch["dialogue"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    targets = tokenizer(
        batch["summary"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_data.map(preprocess_function, batched=True)
val_dataset = val_data.map(preprocess_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=1
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# Train
trainer.train()

# Save
model.save_pretrained("finetuned_bart_samsum")
tokenizer.save_pretrained("finetuned_bart_samsum")

logger.info("Fine-tuning complete.")
