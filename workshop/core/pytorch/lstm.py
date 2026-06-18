"""Character-level LSTM Text Generation in PyTorch."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Corpus to train on (Alan Turing quote)
CORPUS = "we can only see a short distance ahead, but we can see plenty there that needs to be done."


# Model Definition
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        # We only care about the output of the last sequence step
        out = self.fc(out[:, -1, :])
        return out, hidden


def main(hook=None, config=None):
    from workshop.utils.hooks import NoOpProgressHook

    config = config or {}
    hook = hook or NoOpProgressHook()

    epochs = int(config.get("epochs", 10))
    hidden_dim = int(config.get("hidden_dim", 64))
    temperature = float(config.get("temperature", 0.7))
    seq_len = 10
    embedding_dim = 16

    print("Character-level LSTM Text Generation")
    print("=" * 45)
    print(f"Corpus size: {len(CORPUS)} characters")
    print(f"Hidden Dim: {hidden_dim}, Temp: {temperature}")
    print(f"Epochs: {epochs}")
    print()

    if hook.is_cancelled():
        return
    hook.update_stage("Text Tokenization", 10)
    print("Tokenizing character vocabulary...")

    # Vocabulary setup
    chars = sorted(list(set(CORPUS)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    print(f"Vocabulary size: {vocab_size} unique characters.")

    # Create sequences: input length seq_len, target is the 1-step shifted character
    inputs = []
    targets = []
    for i in range(len(CORPUS) - seq_len):
        seq_in = CORPUS[i : i + seq_len]
        char_out = CORPUS[i + seq_len]
        inputs.append([char_to_ix[ch] for ch in seq_in])
        targets.append(char_to_ix[char_out])

    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)

    if hook.is_cancelled():
        return
    hook.update_stage("Model Setup", 20)

    model = CharLSTM(vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print("LSTM model ready.")

    if hook.is_cancelled():
        return
    hook.update_stage("Training", 30)

    reporting_interval = max(1, epochs // 10)

    for epoch in range(epochs):
        if hook.is_cancelled():
            return

        model.train()
        optimizer.zero_grad()

        # Batch size is the entire small dataset for simplicity
        inputs_device = X.to(device)
        targets_device = Y.to(device)

        outputs, _ = model(inputs_device)
        loss = criterion(outputs, targets_device)
        loss.backward()
        optimizer.step()

        perplexity = np.exp(loss.item())

        # Update metrics
        if epoch % reporting_interval == 0 or epoch == epochs - 1:
            progress = 30 + int(50 * ((epoch + 1) / epochs))
            hook.update_stage("Training", progress)
            hook.update_metrics({"epoch": epoch + 1, "loss": float(loss.item()), "perplexity": float(perplexity)})
            print(f"Epoch {epoch + 1:2d}/{epochs}: Loss={loss.item():.4f}, Perplexity={perplexity:.2f}")

    if hook.is_cancelled():
        return
    hook.update_stage("Sampling Text", 85)
    print("\nGenerating character generation probabilities from seed...")

    # Seed string: "we can see"
    seed = "we can see"
    seed_encoded = [char_to_ix[ch] for ch in seed if ch in char_to_ix]
    # Ensure seed matches seq_len by padding/slicing
    if len(seed_encoded) < seq_len:
        # Pad left with spaces (encoded index for space if exists)
        space_ix = char_to_ix.get(" ", 0)
        seed_encoded = [space_ix] * (seq_len - len(seed_encoded)) + seed_encoded
    else:
        seed_encoded = seed_encoded[-seq_len:]

    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([seed_encoded], dtype=torch.long).to(device)
        logits, _ = model(input_tensor)

        # Apply temperature scaling
        scaled_logits = logits[0] / max(0.01, temperature)
        probs = torch.softmax(scaled_logits, dim=0).cpu().numpy()

    # Get top 8 predicted characters
    top_indices = np.argsort(probs)[-8:][::-1]
    top_probs = probs[top_indices]
    top_chars = [repr(ix_to_char[idx]) for idx in top_indices]  # repr to show space as ' '

    if hook.is_cancelled():
        return
    hook.update_stage("Visualization", 90)

    # Plot token probabilities bar chart
    plt.figure(figsize=(6, 5))
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(top_probs)))
    plt.bar(top_chars, top_probs, color=colors, edgecolor="black", alpha=0.8)
    plt.ylabel("Probability")
    plt.xlabel("Predicted Character Token")
    plt.title(f"LSTM Next-Token Probabilities (Seed: '{seed}')")
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    hook.save_plot("lstm_token_probabilities.png", dpi=150, bbox_inches="tight")
    plt.close()

    hook.update_stage("Complete", 100)
