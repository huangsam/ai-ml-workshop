from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from workshop.utils import get_device

# --- 1. CONFIGURATION CONSTANTS ---
MODEL_NAME = "bert-base-uncased"  # The Hugging Face model to use
MAX_LENGTH = 128  # Max length for tokenization
BATCH_SIZE = 16  # Batch size for training
DEVICE = "cpu"  # Default to CPU; Mac M3 will often auto-accelerate PyTorch


def load_data(dataset_name="imdb") -> DatasetDict:
    """
    Loads and prepares the dataset from the Hugging Face Hub.
    """
    print(f"Loading dataset: {dataset_name}...")
    # Load the train and test split for the chosen dataset
    dataset: DatasetDict = load_dataset(dataset_name)
    return dataset


def preprocess_function(examples, tokenizer: AutoTokenizer):
    """
    Tokenization function to preprocess the text data.
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
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
        num_batches: int = len(train_loader)
        # Iterate over batches from the training DataLoader
        for batch_idx, batch in enumerate(train_loader):
            # Batch items are already torch tensors; move to device
            input_ids: torch.Tensor = batch["input_ids"].to(device)
            attention_mask: torch.Tensor = batch["attention_mask"].to(device)
            labels: torch.Tensor = batch["label"].to(device)

            # Forward pass: compute model outputs and loss
            outputs: Any = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss: torch.Tensor = outputs.loss  # Extract the loss value from outputs

            optimizer.zero_grad()  # Reset gradients from previous step
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update model weights

            total_loss += loss.item()  # Accumulate batch loss

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
                print(f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.4f}")
        avg_loss: float = total_loss / num_batches  # Average loss for the epoch
        print(f"Epoch {epoch + 1}/{epochs} - Training loss: {avg_loss:.4f}")

    return avg_loss


def evaluate_model(model: AutoModelForSequenceClassification, test_loader: torch.utils.data.DataLoader, device: str) -> float:
    """
    Evaluate the model on test data.

    Args:
        model: The BERT model for sequence classification
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        Test accuracy
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    correct: int = 0
    total: int = 0
    # Disable gradient calculation for evaluation (faster, less memory)
    with torch.no_grad():
        for batch in test_loader:
            # Batch items are already torch tensors; move to device
            input_ids: torch.Tensor = batch["input_ids"].to(device)
            attention_mask: torch.Tensor = batch["attention_mask"].to(device)
            labels: torch.Tensor = batch["label"].to(device)
            # Forward pass (no labels needed for prediction)
            outputs: Any = model(input_ids=input_ids, attention_mask=attention_mask)
            preds: torch.Tensor = torch.argmax(outputs.logits, dim=1)  # Get predicted class
            correct += (preds == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total samples
    accuracy: float = correct / total if total > 0 else 0  # Compute accuracy
    return accuracy


def main():
    """
    Main entry point for the text classification project.
    """
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
    tokenized_datasets: DatasetDict = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    print("Data tokenization complete.")
    print(tokenized_datasets)

    # 3. Model Loading (to ensure environment works)
    model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)  # Move the model to the chosen device (CPU or MPS)

    print("\nSetup complete. Starting training...")

    # 4. Prepare DataLoaders
    from torch.utils.data import DataLoader

    train_dataset: Dataset = tokenized_datasets["train"].with_format("torch")
    test_dataset: Dataset = tokenized_datasets["test"].with_format("torch")
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 5. Optimizer
    optimizer: torch.optim.AdamW = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 6. Training
    EPOCHS: int = 2
    final_train_loss = train_model(model, train_loader, optimizer, DEVICE, EPOCHS)
    print(f"Final Training Loss: {final_train_loss:.4f}")

    # 7. Evaluation
    test_accuracy = evaluate_model(model, test_loader, DEVICE)
    print(f"Test Accuracy: {test_accuracy:.4f}")

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
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            label = "Positive" if prediction == 1 else "Negative"
            print(f'Text: "{text[:50]}..."')
            print(f"Prediction: {label}\n")


if __name__ == "__main__":
    main()
