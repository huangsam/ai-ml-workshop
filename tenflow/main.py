from typing import Any

import tensorflow as tf
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# --- 1. CONFIGURATION CONSTANTS ---
MODEL_NAME = "bert-base-uncased"  # The Hugging Face model to use
MAX_LENGTH = 128  # Max length for tokenization
BATCH_SIZE = 16  # Batch size for training
EPOCHS = 2  # Number of training epochs


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


def main():
    """
    Main entry point for the text classification project using TensorFlow/Keras.
    """
    # 1. Check for GPU/TPU availability (TensorFlow handles acceleration automatically)
    print("TensorFlow version:", tf.__version__)
    if tf.config.list_physical_devices("GPU"):
        print("🔥 GPU available. Using GPU for acceleration.")
    else:
        print("Using CPU.")

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

    # 3. Model Loading
    model: TFAutoModelForSequenceClassification = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 4. Prepare tf.data Datasets using transformers built-in method
    train_dataset = tokenized_datasets["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        shuffle=True,
        batch_size=BATCH_SIZE,
    )
    test_dataset = tokenized_datasets["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        shuffle=False,
        batch_size=BATCH_SIZE,
    )

    # 5. Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print("\nSetup complete. Starting training...")

    # 6. Training
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, verbose=1)

    # 7. Evaluation
    print("\nEvaluating on test set...")
    results = model.evaluate(test_dataset, verbose=1)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")


if __name__ == "__main__":
    main()
