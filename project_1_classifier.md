# Project 1: Custom Text Classifier (PyTorch/TensorFlow)

## 🎯 Goal
Train a model to classify text (e.g., movie reviews, social media posts) into two categories (e.g., POSITIVE/NEGATIVE).

## 🛠️ Technology Stack
* **Backend:** PyTorch (recommended for initial setup) or TensorFlow/Keras
* **Data/Tokenization:** Hugging Face `transformers` and `datasets`

## ⚙️ Steps & Milestones

### 1. Setup & Data Loading
1.  **Environment:** Create a dedicated virtual environment (e.g., `uv venv .venv_torch`).
2.  **Packages:** Install `torch`, `transformers`, `datasets`, and `scikit-learn`.
3.  **Data:** Load a public dataset (e.g., `imdb` or `glue/sst2`) using `datasets.load_dataset()`.

### 2. Tokenization (Hugging Face)
1.  **Tokenizer:** Load an AutoTokenizer (e.g., `'bert-base-uncased'`) using `AutoTokenizer.from_pretrained()`.
2.  **Preprocessing:** Write a function to tokenize the text and pad/truncate sequences to a fixed length (e.g., 128 tokens).
3.  **Mapping:** Apply the tokenization function to the entire dataset using `dataset.map()`.

### 3. PyTorch Data Pipeline
1.  **Dataset:** Convert the tokenized dataset columns (input IDs, attention mask, labels) into PyTorch tensors.
2.  **DataLoader:** Wrap the datasets in a **`torch.utils.data.DataLoader`** for efficient batching and shuffling.

### 4. Model Definition & Training
1.  **Model:** Load a pre-trained model with a classification head, such as `AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)`.
2.  **Training Loop:** Write a basic PyTorch training loop (or use the Keras `model.fit()` API if using TensorFlow).
    * Forward pass.
    * Calculate loss.
    * Backpropagation (`loss.backward()`).
    * Optimizer step.
3.  **Evaluation:** Evaluate the model on the test set and report accuracy.

### 💡 Key Learning Points
* Understanding the role of the **Tokenizer** (text to numbers).
* Mastering the **`Dataset`** and **`DataLoader`** architecture.
* How to load and customize a **pre-trained Hugging Face model** for a new task.
