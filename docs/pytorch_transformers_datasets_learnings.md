# Learnings: Datasets, PyTorch, and Transformers

## Datasets (Hugging Face)
- Provide ready-to-use, standardized datasets for NLP and ML tasks.
- Include labels for supervised learning (e.g., IMDB sentiment: 0=negative, 1=positive).
- Support easy train/test splits and preprocessing with `.map()` and `.select()`.
- Can be converted to PyTorch tensors with `.with_format('torch')` for seamless integration.

## PyTorch
- Deep learning framework for building and training neural networks.
- Uses forward pass (prediction) and backpropagation (learning) for model training.
- Supports GPU/MPS acceleration for faster computation.
- DataLoader enables efficient batching and shuffling of data.
- Ideal for neural networks, not traditional ML algorithms (use scikit-learn for those).

## Transformers (Hugging Face)
- Pretrained models for NLP tasks (e.g., BERT, GPT) with easy fine-tuning.
- `AutoTokenizer` handles text preprocessing and tokenization.
- `AutoModelForSequenceClassification` provides a ready-to-train classification head.
- Passing `labels` to the model automatically computes loss for supervised learning.
- Transformers excel at complex language tasks, but require significant compute resources.

## Practical Integration
- Datasets, PyTorch, and Transformers work together for efficient supervised NLP workflows.
- Preprocessing, batching, training, and evaluation are streamlined with these libraries.
- The exercise demonstrated binary sentiment classification using IMDB, BERT, and PyTorch.
