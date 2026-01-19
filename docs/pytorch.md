# Learnings: PyTorch Deep Learning Across Domains

## PyTorch Fundamentals
- Deep learning framework for building and training neural networks.
- Uses forward pass (prediction) and backpropagation (learning) for model training.
- Supports GPU/MPS acceleration (Apple Silicon: `torch.backends.mps.is_available()`).
- `nn.Module` is the base class for all neural network modules; override `forward()` for predictions.
- `DataLoader` enables efficient batching, shuffling, and parallel data loading.
- Ideal for neural networks, not traditional ML algorithms (use scikit-learn for those).
- Type hints and comprehensive comments in code prevent bugs and aid readability.

## Model Architectures

### Convolutional Neural Networks (CNN)
- **Use case**: Image classification, computer vision tasks
- **Architecture**: Conv2d layers → ReLU activation → MaxPool2d for spatial reduction → Flatten → Dense layers
- **Example**: ResNet-18 on CIFAR-10 achieves ~80% accuracy with transfer learning
- **Data augmentation**: RandomHorizontalFlip, RandomRotation, ColorJitter improve generalization
- **Key metric**: Track both loss and accuracy; use confusion matrix for detailed error analysis

### Recurrent Neural Networks (RNN/LSTM)
- **Use case**: Sequential data (time series, text, temporal patterns)
- **LSTM advantages**: Mitigates vanishing gradient problem; captures long-term dependencies
- **Time series forecasting**: Can predict multiple future steps (sequence-to-sequence)
- **Shape handling**: LSTM outputs (batch_size, seq_length, hidden_size); reshape for final predictions
- **Sequence length matters**: Longer lookback windows capture more temporal context but increase computation

### Transformer Models (BERT)
- **Use case**: NLP tasks (text classification, QA, named entity recognition)
- **BERT strengths**: Bidirectional context, pre-trained on massive text corpora
- **Fine-tuning**: Transfer learning is efficient; often needs only 1-2 epochs
- **Tokenization**: `AutoTokenizer` handles subword tokenization; max_length pads/truncates sequences
- **Task-specific heads**: `AutoModelForSequenceClassification` (classification), `BertForQuestionAnswering` (QA)
- **Warning**: Naive initialization messages can be ignored; model still learns effectively

### Multi-Layer Perceptrons (MLP)
- **Use case**: Tabular/structured data classification
- **Categorical embeddings**: Convert categorical features to dense vectors (embedding_dim << vocab_size)
- **Feature fusion**: Concatenate numerical features with embedded categorical features
- **Architecture**: Sequential stacking of Linear → ReLU → Dropout layers prevents overfitting
- **Dropout**: Essential for tabular data to prevent overfitting; typical rates: 0.2-0.5

## Datasets (Hugging Face)
- Provide ready-to-use, standardized datasets across domains (NLP, vision, tabular).
- Include pre-defined splits (train/test/validation) and labels for supervised learning.
- Support efficient preprocessing with `.map()` method for batched transformations.
- Can be converted to PyTorch tensors with `.with_format('torch')` for seamless integration.
- **Example datasets**: IMDB (sentiment), SQuAD (QA), CIFAR-10 (vision)

## Training Pipeline Best Practices

### Setup
- Device detection: Check for MPS/GPU availability before moving models to device
- Model initialization: Use pre-trained weights when available (transfer learning)
- Loss functions: MSELoss for regression, CrossEntropyLoss for multi-class, BCEWithLogitsLoss for binary

### Training Loop
1. **Forward pass**: `outputs = model(inputs)`
2. **Loss computation**: `loss = criterion(outputs, targets)`
3. **Backpropagation**: `loss.backward()`
4. **Optimization step**: `optimizer.step()` + `optimizer.zero_grad()`
5. **Validation**: Evaluate on test set with `model.eval()` and `torch.no_grad()`

### Evaluation Metrics
- **Classification**: Accuracy (%), precision, recall, F1-score, confusion matrix
- **Regression**: MSE, RMSE, MAE, R² score
- **NLP-specific**: BLEU (machine translation), Exact Match / F1 (QA), Perplexity (language modeling)
- Track both training and test metrics to detect overfitting

## Practical Integration
- Datasets, PyTorch, and Transformers work together for efficient supervised learning workflows.
- Preprocessing, batching, training, and evaluation are streamlined with these libraries.
- **Cross-domain patterns**:
  - All domains benefit from train/test splits and proper data preprocessing
  - Model selection depends on data type: CNN for images, LSTM for sequences, Transformer for NLP
  - Evaluation metrics vary by domain but share common principles (loss + task-specific metrics)
- **Production readiness**: Type hints, error handling, and modular code structure enable easy deployment

## Fine-Tuning and Parameter-Efficient Methods

### PEFT (Parameter-Efficient Fine-Tuning)
- **Purpose**: Adapt large pre-trained models for downstream tasks without updating all parameters
- **Key benefit**: Enables fine-tuning billion-parameter models on consumer hardware
- **Memory efficiency**: Train only adapter parameters instead of full model weights

### LoRA (Low-Rank Adaptation)
- **Concept**: Inject trainable low-rank matrices into frozen pre-trained weights
- **Architecture**: For each weight matrix W, add ΔW = BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
- **Hyperparameters**:
  - **Rank (r)**: Dimension of low-rank matrices (8-64 typical; lower = fewer parameters)
  - **Alpha (α)**: Scaling factor for LoRA updates (usually 2×r)
  - **Dropout**: Regularization in LoRA layers (0.05-0.1 typical)
- **Target modules**: Model components to adapt (attention layers: query, value, key projections)
  - BERT/DistilBERT: `["q_lin", "v_lin"]` (DistilBERT uses linear projections)
  - GPT/Llama: `["q_proj", "v_proj", "k_proj"]` (standard attention naming)
- **Parameter reduction**: Achieves 95-99% reduction in trainable parameters

### Implementation with Hugging Face
- **Setup**: Use `peft.LoraConfig` to configure adapters, `get_peft_model()` to apply
- **Training**: Standard PyTorch training loop works unchanged; only adapters are updated
- **Inference**: Load base model + adapters, or merge for deployment efficiency
- **Task types**: SEQ_CLS (classification), SEQ_2_SEQ_LM (generation), CAUSAL_LM (language modeling)

### Practical Benefits
- **Hardware requirements**: Fine-tune Llama-7B on single GPU with 8GB VRAM
- **Training speed**: 2-5x faster than full fine-tuning
- **Multi-task adaptation**: Train separate LoRA adapters for different tasks
- **Memory overhead**: Minimal (<1% of original model size for adapter storage)
- **Compatibility**: Works with existing training pipelines and optimizers

### Best Practices
- **Rank selection**: Start with r=8-16; increase for complex tasks
- **Target modules**: Focus on attention layers; include MLP layers for generation tasks
- **Learning rate**: Use 1e-4 to 5e-4 (higher than full fine-tuning)
- **Dataset size**: Effective with 100-1000 samples; benefits from quality over quantity
- **Evaluation**: Monitor both adapter performance and generalization to new tasks
