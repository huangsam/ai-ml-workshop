"""
Parameter-Efficient Fine-Tuning with PEFT/LoRA

This script demonstrates fine-tuning a pre-trained model efficiently using LoRA (Low-Rank Adaptation).
It shows how to adapt large models for downstream tasks without updating all parameters.

Dataset: Custom domain-specific text classification (using legal/financial/technical text)
Model: DistilBERT with LoRA adapters
Key concepts: LoRA, PEFT, parameter efficiency, adapter-based fine-tuning
"""

from typing import Tuple

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from workshop.utils import get_device

# --- 1. CONFIGURATION CONSTANTS ---
MODEL_NAME = "distilbert-base-uncased"  # Lightweight baseline
TASK_TYPE = TaskType.SEQ_CLS  # Sequence classification
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
DEVICE = "cpu"  # Will check for MPS acceleration

# LoRA Configuration
LORA_R = 8  # Rank of LoRA update matrices (lower = fewer parameters)
LORA_ALPHA = 16  # LoRA scaling factor
LORA_DROPOUT = 0.05  # Dropout in LoRA layers
TARGET_MODULES = ["q_lin", "v_lin"]  # DistilBERT attention modules (query, value)


def load_data(dataset_name: str = "ag_news", subset_size: int = 1000) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare a dataset for fine-tuning.

    Args:
        dataset_name: Hugging Face dataset name
        subset_size: Number of samples to use (for faster training)

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    print(f"Loading dataset: {dataset_name}...")

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Use subset for faster training
    if subset_size < len(dataset["train"]):
        train_dataset = dataset["train"].select(range(subset_size))
        test_dataset = dataset["test"].select(range(min(subset_size // 4, len(dataset["test"]))))
    else:
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset


def preprocess_data(train_dataset: Dataset, test_dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Preprocess and tokenize the data.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
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
        return tokenizer(examples[text_column], padding="max_length", truncation=True, max_length=max_length)

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True, desc="Tokenizing train")
    test_dataset = test_dataset.map(tokenize_function, batched=True, desc="Tokenizing test")

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

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
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The PEFT model with LoRA
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

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
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"Batch {batch_idx + 1} | Loss: {loss.item():.4f}")

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy


def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float]:
    """
    Evaluate the model on test data.

    Args:
        model: The PEFT model with LoRA
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            predictions = outputs.logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
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

    # 1. Device setup
    global DEVICE
    DEVICE = get_device()
    if DEVICE == "mps":
        print(f"🔥 Found MPS device. Using {DEVICE} for acceleration.")
    else:
        print("Using CPU for computation.")

    # 2. Load data
    train_dataset, test_dataset = load_data(dataset_name="ag_news", subset_size=1000)

    # 3. Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. Prepare data loaders
    train_loader, test_loader = preprocess_data(train_dataset, test_dataset, tokenizer)

    # 5. Setup LoRA model
    num_labels = train_dataset.features["label"].num_classes
    model = setup_lora_model(MODEL_NAME, num_labels)
    model.to(DEVICE)

    # 6. Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
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
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step()

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

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
    print("\nKey Benefits of LoRA:")
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
