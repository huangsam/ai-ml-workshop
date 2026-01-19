"""
Question Answering with PyTorch

This script demonstrates extractive question answering using SQuAD dataset and BERT.
It covers NLP fundamentals: span prediction, attention mechanisms, and QA evaluation.

Dataset: SQuAD v1.1 (extractive QA on Wikipedia articles)
Model: BERT-base for question answering
"""

from typing import Dict, List, Tuple

import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, DefaultDataCollator

from utils import get_device

# --- 1. CONFIGURATION CONSTANTS ---
MODEL_NAME = "bert-base-uncased"  # The Hugging Face model to use
MAX_LENGTH = 384  # Max length for question + context
DOC_STRIDE = 128  # Stride for sliding window when context is long
BATCH_SIZE = 8  # Smaller batch size due to longer sequences
DEVICE = "cpu"  # Default to CPU; will check for MPS acceleration


def load_data(dataset_name: str = "squad") -> DatasetDict:
    """
    Loads and prepares the SQuAD dataset from the Hugging Face Hub.

    Args:
        dataset_name: Name of the dataset to load

    Returns:
        DatasetDict containing train/validation splits
    """
    print(f"Loading dataset: {dataset_name}...")
    dataset: DatasetDict = load_dataset(dataset_name)
    return dataset


def preprocess_function(examples: Dict[str, List], tokenizer: AutoTokenizer) -> Dict[str, List]:
    """
    Tokenization function to preprocess question-answering data.

    Args:
        examples: Batch of examples from dataset
        tokenizer: The tokenizer to use

    Returns:
        Tokenized examples with start/end positions
    """
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    # Tokenize question + context pairs
    inputs = tokenizer(
        questions,
        contexts,
        max_length=MAX_LENGTH,
        truncation="only_second",  # Truncate context, not question
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,  # Handle long contexts with sliding window
        return_offsets_mapping=True,  # Get character positions for answer mapping
        padding="max_length",
    )

    # Map answer positions to token positions
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        # If no answer, use CLS token position
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Get start/end character positions
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            # Find token positions
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:  # 1 = context, 0 = question
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If answer is not in context, use CLS
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Find token positions for answer
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def compute_metrics(eval_pred: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
    """
    Computes F1 and Exact Match scores for question answering.

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary with F1 and EM scores
    """
    predictions, labels = eval_pred

    # Get predicted start/end positions
    start_preds = predictions[0].argmax(dim=-1)
    end_preds = predictions[1].argmax(dim=-1)

    # Simple accuracy for demonstration (real QA uses F1/EM)
    start_acc = (start_preds == labels[0]).float().mean()
    end_acc = (end_preds == labels[1]).float().mean()

    return {
        "start_accuracy": start_acc.item(),
        "end_accuracy": end_acc.item(),
    }


def main() -> None:
    """
    Main entry point for the question answering project.
    """
    print("Question Answering with SQuAD and BERT")
    print("=" * 45)

    # 1. Device setup
    global DEVICE
    DEVICE = get_device()
    if DEVICE == "mps":
        print(f"🔥 Found MPS device. Using {DEVICE} for acceleration.")
    else:
        print("Using CPU for computation.")

    # 2. Load data and tokenizer
    raw_datasets: DatasetDict = load_data()
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Take a small subset for demonstration
    small_train_dataset = raw_datasets["train"].select(range(1000))
    small_eval_dataset = raw_datasets["validation"].select(range(200))

    print(f"Training samples: {len(small_train_dataset)}")
    print(f"Validation samples: {len(small_eval_dataset)}")

    # 3. Preprocess data
    print("\nPreprocessing data...")
    tokenized_train = small_train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=small_train_dataset.column_names,
    )
    tokenized_eval = small_eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=small_eval_dataset.column_names,
    )

    print("Data preprocessing complete.")

    # 4. Create model
    model: AutoModelForQuestionAnswering = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    # 5. Create data collator and dataloaders
    data_collator = DefaultDataCollator()

    from torch.utils.data import DataLoader

    train_loader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    eval_loader = DataLoader(tokenized_eval, batch_size=BATCH_SIZE, collate_fn=data_collator)

    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 7. Training loop
    print("\nStarting training...")
    NUM_EPOCHS = 2

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training loss: {avg_loss:.4f}")

    # 8. Evaluation
    print("\nEvaluating model...")
    model.eval()
    eval_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            eval_loss += outputs.loss.item()
            num_batches += 1

    avg_eval_loss = eval_loss / num_batches
    print(f"Validation loss: {avg_eval_loss:.4f}")

    print("\nTraining complete! 🎉")
    print("Note: For full QA evaluation, use proper F1/Exact Match metrics.")


if __name__ == "__main__":
    main()
