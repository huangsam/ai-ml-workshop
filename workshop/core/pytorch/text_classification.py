import random
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from workshop.utils import get_device

# --- 1. CONFIGURATION CONSTANTS ---
MODEL_NAME = "bert-base-uncased"  # The Hugging Face model to use
MAX_LENGTH = 128  # Max length for tokenization
BATCH_SIZE = 16  # Batch size for training
DEVICE = "cpu"  # Default to CPU; Mac M3 will often auto-accelerate PyTorch
SEED = 42


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(dataset_name="imdb") -> DatasetDict:
    """
    Loads and prepares the dataset from the Hugging Face Hub.
    """
    print(f"Loading dataset: {dataset_name}...")
    # Load the train and test split for the chosen dataset
    dataset: DatasetDict = load_dataset(dataset_name)

    # For demonstration purposes, use a smaller subset to speed up training
    # IMDB is large (25k train, 25k test), so let's use 2k for this workshop
    print("Subsampling dataset for workshop speed...")
    small_train_dataset = dataset["train"].shuffle(seed=SEED).select(range(2000))
    small_test_dataset = dataset["test"].shuffle(seed=SEED).select(range(500))

    return DatasetDict({"train": small_train_dataset, "test": small_test_dataset})


def preprocess_function(examples, tokenizer: AutoTokenizer):
    """
    Tokenization function to preprocess the text data.
    """
    # We truncate here, but padding will be handled dynamically by the DataCollator
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def train_model(
    model: AutoModelForSequenceClassification,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.AdamW,
    device: str,
    epochs: int,
) -> float:
    """
    Train the model for the specified number of epochs.

    Args:
        model: The BERT model for sequence classification
        train_loader: Training data loader
        optimizer: Optimizer for updating model parameters
        device: Device to run training on
        epochs: Number of training epochs

    Returns:
        Average training loss from the last epoch
    """
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss: float = 0.0
        # Use tqdm for a nice progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            # Batch items are already torch tensors; move to device
            input_ids: torch.Tensor = batch["input_ids"].to(device)
            attention_mask: torch.Tensor = batch["attention_mask"].to(device)
            labels: torch.Tensor = batch["labels"].to(device)

            # Forward pass: compute model outputs and loss
            outputs: Any = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss: torch.Tensor = outputs.loss  # Extract the loss value from outputs

            optimizer.zero_grad()  # Reset gradients from previous step
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update model weights

            total_loss += loss.item()  # Accumulate batch loss

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss: float = total_loss / len(train_loader)  # Average loss for the epoch
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

    return avg_loss


def evaluate_model(model: AutoModelForSequenceClassification, test_loader: torch.utils.data.DataLoader, device: str) -> dict:
    """
    Evaluate the model on test data using sklearn metrics.

    Args:
        model: The BERT model for sequence classification
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1)
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    all_preds = []
    all_labels = []

    # Disable gradient calculation for evaluation (faster, less memory)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Batch items are already torch tensors; move to device
            input_ids: torch.Tensor = batch["input_ids"].to(device)
            attention_mask: torch.Tensor = batch["attention_mask"].to(device)
            labels: torch.Tensor = batch["labels"].to(device)

            # Forward pass (no labels needed for prediction, but we pass them for consistency if needed)
            outputs: Any = model(input_ids=input_ids, attention_mask=attention_mask)
            preds: torch.Tensor = torch.argmax(outputs.logits, dim=1)  # Get predicted class

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics using sklearn
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    """
    Main entry point for the text classification project.
    """
    # 0. Set seed for reproducibility
    set_seed(SEED)

    # 1. Check for Mac Metal Performance Shaders (MPS) for M-series acceleration
    # This is an important step to leverage your M3 Max chip
    global DEVICE
    DEVICE = get_device()
    if DEVICE == "mps":
        print(f"🔥 Found MPS device. Using {DEVICE} for acceleration.")
    else:
        print(f"Using {DEVICE.upper()}.")

    # 2. Load Data and Tokenizer
    raw_datasets: DatasetDict = load_data()

    # Instantiate the tokenizer (needed for all text processing)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize datasets
    # Note: We don't pad here anymore! DataCollator will do it dynamically.
    tokenized_datasets: DatasetDict = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
    )
    # We need to remove the text column as the model doesn't accept it
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # Rename 'label' to 'labels' as expected by Hugging Face models
    if "label" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    print("Data tokenization complete.")
    print(tokenized_datasets)

    # 3. Model Loading (to ensure environment works)
    model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)  # Move the model to the chosen device (CPU or MPS)

    print("\nSetup complete. Starting training...")

    # 4. Prepare DataLoaders
    from torch.utils.data import DataLoader

    # DataCollatorWithPadding improves efficiency by padding to the longest sequence IN THE BATCH
    # rather than the longest sequence in the whole dataset or MAX_LENGTH
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader: DataLoader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    test_loader: DataLoader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE, collate_fn=data_collator)

    # 5. Optimizer
    optimizer: torch.optim.AdamW = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 6. Training
    EPOCHS: int = 2
    final_train_loss = train_model(model, train_loader, optimizer, DEVICE, EPOCHS)
    print(f"Final Training Loss: {final_train_loss:.4f}")

    # 7. Evaluation
    metrics = evaluate_model(model, test_loader, DEVICE)
    print("\nModel Evaluation:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

    # 8. Prediction demo
    print("\n" + "=" * 50)
    print("Prediction Demo")
    print("=" * 50)
    sample_texts = [
        "This movie was absolutely fantastic! I loved every moment of it.",
        "Terrible film. Complete waste of time and money.",
    ]
    model.eval()
    with torch.no_grad():
        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            label = "Positive" if prediction == 1 else "Negative"
            print(f'Text: "{text[:50]}..."')
            print(f"Prediction: {label}\n")


if __name__ == "__main__":
    main()
