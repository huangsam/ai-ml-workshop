"""
Parameter-Efficient Fine-Tuning with PEFT/LoRA

This script demonstrates fine-tuning a pre-trained model efficiently using LoRA (Low-Rank Adaptation).
It shows how to adapt large models for downstream tasks without updating all parameters.

Dataset: Custom domain-specific text classification (using legal/financial/technical text)
Model: DistilBERT with LoRA adapters
Key concepts: LoRA, PEFT, parameter efficiency, adapter-based fine-tuning
"""

import random
import textwrap

import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from workshop.utils import get_device

# --- 1. CONFIGURATION CONSTANTS ---
MODEL_NAME = "distilbert-base-uncased"  # Lightweight baseline
TASK_TYPE = TaskType.SEQ_CLS  # Sequence classification
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
DEVICE = "cpu"  # Will check for MPS acceleration
SEED = 42

# LoRA Configuration
LORA_R = 8  # Rank of LoRA update matrices (lower = fewer parameters)
LORA_ALPHA = 16  # LoRA scaling factor
LORA_DROPOUT = 0.05  # Dropout in LoRA layers
TARGET_MODULES = ["q_lin", "v_lin"]  # DistilBERT attention modules (query, value)


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(dataset_name: str = "ag_news", subset_size: int = 1000) -> DatasetDict:
    """
    Load and prepare a dataset for fine-tuning.

    Args:
        dataset_name: Hugging Face dataset name
        subset_size: Number of samples to use (for faster training)

    Returns:
        DatasetDict containing train/test splits
    """
    print(f"Loading dataset: {dataset_name}...")

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Use subset for faster training
    if subset_size < len(dataset["train"]):
        train_dataset = dataset["train"].shuffle(seed=SEED).select(range(subset_size))
        test_dataset = dataset["test"].shuffle(seed=SEED).select(range(min(subset_size // 4, len(dataset["test"]))))
    else:
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return DatasetDict({"train": train_dataset, "test": test_dataset})


def preprocess_data(dataset_dict: DatasetDict, tokenizer: AutoTokenizer, max_length: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    Preprocess and tokenize the data.

    Args:
        dataset_dict: DatasetDict containing splits
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length

    Returns:
        Tuple of (train_loader, test_loader)
    """
    print("Preprocessing data...")

    def tokenize_function(examples):
        """Tokenize text data."""
        # Handle dataset column names (varies by dataset)
        text_column = "text" if "text" in examples.keys() else "content"
        # Truncate here, allow DataCollator to pad
        return tokenizer(examples[text_column], truncation=True, max_length=max_length)

    # Tokenize datasets using .map() on the DatasetDict
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, desc="Tokenizing")

    # Rename label to labels if needed
    if "label" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Set format to remove unused columns but keep tensors
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    tokenized_datasets.set_format(type="torch", columns=columns_to_keep)

    # Use DataCollatorWithPadding for dynamic padding (efficiency)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create data loaders
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    test_loader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

    return train_loader, test_loader


def setup_lora_model(model_name: str, num_labels: int) -> nn.Module:
    """
    Load a pre-trained model and apply LoRA adapters.

    Args:
        model_name: Hugging Face model name
        num_labels: Number of classification labels

    Returns:
        Model with LoRA adapters
    """
    print(f"Loading base model: {model_name}...")

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {original_params:,}")

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TASK_TYPE,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Count parameters after LoRA
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    percent_trainable = 100 * trainable_params / total_params
    param_reduction = (1 - trainable_params / original_params) * 100

    print("\nLoRA Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {percent_trainable:.2f}%")
    print(f"Parameter reduction: {param_reduction:.2f}%")
    print(f"\nLoRA config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")

    # Store for later use
    model._param_reduction = param_reduction

    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The PEFT model with LoRA
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to run on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        predictions = outputs.logits.argmax(dim=1)
        correct += torch.sum(predictions == labels).item()
        total += labels.size(0)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy


def evaluate(model: nn.Module, test_loader: DataLoader, device: str) -> tuple[float, float]:
    """
    Evaluate the model on test data.

    Args:
        model: The PEFT model with LoRA
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(test_loader, desc="Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            predictions = outputs.logits.argmax(dim=1)
            correct += torch.sum(predictions == labels).item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)

    return avg_loss, accuracy


def main() -> None:
    """
    Main entry point for fine-tuning with LoRA.
    """
    print("=" * 60)
    print("Parameter-Efficient Fine-Tuning with LoRA")
    print("=" * 60)

    # 0. Set seed
    set_seed(SEED)

    # 1. Device setup
    global DEVICE
    DEVICE = get_device()
    if DEVICE == "mps":
        print(f"🔥 Found MPS device. Using {DEVICE} for acceleration.")
    else:
        print("Using CPU for computation.")

    # 2. Load data
    dataset_dict = load_data(dataset_name="ag_news", subset_size=1000)

    # 3. Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. Prepare data loaders
    train_loader, test_loader = preprocess_data(dataset_dict, tokenizer)

    # 5. Setup LoRA model
    num_labels = 4  # AG News has 4 classes
    model = setup_lora_model(MODEL_NAME, num_labels)
    model.to(DEVICE)

    # 6. Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 7. Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, DEVICE)
        scheduler.step()

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, DEVICE)

        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Testing  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            # Could save model here: model.save_pretrained('best_lora_model')

    # 8. Final results
    print("\n" + "=" * 60)
    print("Training complete! 🎉")
    print("=" * 60)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")

    # 9. Prediction demo
    print("\n" + "=" * 60)
    print("Prediction Demo")
    print("=" * 60)
    # AG News categories: World, Sports, Business, Sci/Tech
    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    sample_texts = [
        "The stock market reached record highs today as investors celebrated.",
        "The team won the championship after a thrilling overtime victory.",
        "Scientists discovered a new exoplanet in a nearby solar system.",
    ]
    model.eval()
    with torch.no_grad():
        for text in sample_texts:
            # Type hint helps linter understand text is str
            text: str = text
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            short_text = textwrap.shorten(text, width=60, placeholder="...")
            print(f'Text: "{short_text}"')
            print(f"Category: {label_names[prediction]}\n")

    print("Key Benefits of LoRA:")
    print(f"  • {model._param_reduction:.2f}% fewer parameters to train")
    print("  • Faster training on consumer hardware (M3 Mac or single GPU)")
    print("  • Easy to adapt to multiple downstream tasks")
    print("  • Minimal memory overhead")
    print("\nNext steps:")
    print("  1. Save LoRA adapters: model.save_pretrained('my_lora_adapters')")
    print("  2. Load adapters in inference: model = PeftModel.from_pretrained(model, 'my_lora_adapters')")
    print("  3. Merge for deployment: model.merge_and_unload()")


if __name__ == "__main__":
    main()
