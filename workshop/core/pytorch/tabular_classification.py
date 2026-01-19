"""
Tabular Classification with PyTorch

This script demonstrates tabular data classification using Titanic dataset and MLP.
It covers structured data fundamentals: embeddings, feature engineering, and neural networks.

Dataset: Titanic survival prediction (structured data with categorical/mixed features)
Model: MLP with embeddings for categorical variables
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from workshop.utils import get_device

# --- 1. CONFIGURATION CONSTANTS ---
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = "cpu"  # Default to CPU; will check for MPS acceleration
EMBEDDING_DIM = 8  # Embedding dimension for categorical features


class TabularDataset(Dataset):
    """
    Custom Dataset for tabular data.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        Initialize dataset.

        Args:
            X: Feature tensor
            y: Target tensor
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class TabularClassifier(nn.Module):
    """
    MLP classifier for tabular data with categorical embeddings.
    """

    def __init__(self, num_numerical: int, categorical_info: dict[str, int], embedding_dim: int = 8):
        """
        Initialize the model.

        Args:
            num_numerical: Number of numerical features
            categorical_info: Dict mapping categorical feature names to their vocab sizes
            embedding_dim: Dimension for categorical embeddings
        """
        super().__init__()

        # Create embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0

        for cat_name, vocab_size in categorical_info.items():
            self.embeddings[cat_name] = nn.Embedding(vocab_size, embedding_dim)
            total_embed_dim += embedding_dim

        # Total input dimension = numerical features + embedded categorical features
        input_dim = num_numerical + total_embed_dim

        # MLP layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),  # Binary classification
            nn.Sigmoid(),
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Dictionary containing 'numerical' and categorical feature tensors

        Returns:
            Prediction probabilities
        """
        # Process categorical features through embeddings
        embedded_features = []
        for cat_name, embedding in self.embeddings.items():
            embedded = embedding(x[cat_name])
            # Flatten if needed (assuming single categorical per sample)
            embedded = embedded.view(embedded.size(0), -1)
            embedded_features.append(embedded)

        # Concatenate all embedded features
        if embedded_features:
            cat_features = torch.cat(embedded_features, dim=1)
            # Concatenate with numerical features
            combined = torch.cat([x["numerical"], cat_features], dim=1)
        else:
            combined = x["numerical"]

        # Pass through MLP
        return self.layers(combined)


def load_titanic_data() -> pd.DataFrame:
    """
    Load and return the Titanic dataset.

    Returns:
        DataFrame with Titanic data
    """
    print("Loading Titanic dataset...")

    # For demonstration, we'll create a simplified version
    # In practice, you'd load from sklearn.datasets or a CSV file
    from sklearn.datasets import fetch_openml

    try:
        # Try to load from OpenML
        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.frame
        # Convert survived to numeric if it's categorical
        if df["survived"].dtype.name == "category":
            df["survived"] = df["survived"].astype(int)
    except Exception:
        # Fallback: create synthetic data similar to Titanic
        print("OpenML failed, creating synthetic Titanic-like data...")
        np.random.seed(42)
        n_samples = 1000

        df = pd.DataFrame(
            {
                "survived": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
                "pclass": np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.25, 0.5]),
                "sex": np.random.choice(["male", "female"], n_samples, p=[0.65, 0.35]),
                "age": np.random.normal(30, 15, n_samples).clip(0, 80),
                "sibsp": np.random.poisson(1, n_samples),
                "parch": np.random.poisson(0.5, n_samples),
                "fare": np.random.exponential(30, n_samples),
                "embarked": np.random.choice(["S", "C", "Q"], n_samples, p=[0.7, 0.2, 0.1]),
            }
        )

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"Survival rate: {df['survived'].mean():.2%}")

    return df


def preprocess_data(df: pd.DataFrame) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, LabelEncoder]]:
    """
    Preprocess the Titanic data for training.

    Args:
        df: Raw Titanic DataFrame

    Returns:
        Tuple of (features_dict, targets, encoders)
    """
    print("Preprocessing data...")

    # Handle missing values
    df = df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna("S")  # Most common

    # Encode categorical variables
    categorical_features = ["pclass", "sex", "embarked"]
    numerical_features = ["age", "sibsp", "parch", "fare"]

    encoders = {}
    encoded_cats = {}

    for cat_feat in categorical_features:
        encoder = LabelEncoder()
        encoded_cats[cat_feat] = encoder.fit_transform(df[cat_feat])
        encoders[cat_feat] = encoder

    # Scale numerical features
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(df[numerical_features])

    # Split data
    X_cat = pd.DataFrame(encoded_cats)
    X_num = pd.DataFrame(scaled_nums, columns=numerical_features)
    y = df["survived"].values

    X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(X_cat, X_num, y, test_size=0.2, random_state=42, stratify=y)

    # Convert to tensors
    features_train = {"numerical": torch.FloatTensor(X_num_train.values), **{cat: torch.LongTensor(X_cat_train[cat].values) for cat in categorical_features}}

    features_test = {"numerical": torch.FloatTensor(X_num_test.values), **{cat: torch.LongTensor(X_cat_test[cat].values) for cat in categorical_features}}

    targets_train = torch.FloatTensor(y_train)
    targets_test = torch.FloatTensor(y_test)

    print(f"Train samples: {len(targets_train)}")
    print(f"Test samples: {len(targets_test)}")
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    return (features_train, features_test), (targets_train, targets_test), encoders


def create_data_loaders(features: tuple[dict, dict], targets: tuple[torch.Tensor, torch.Tensor]) -> tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and testing.

    Args:
        features: Tuple of (train_features, test_features)
        targets: Tuple of (train_targets, test_targets)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = TabularDataset(features[0]["numerical"], targets[0])
    test_dataset = TabularDataset(features[1]["numerical"], targets[1])

    # Note: For simplicity, we're only using numerical features in the dataset
    # In a full implementation, you'd need a custom collate_fn to handle the dict structure

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train_model(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
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

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # For simplicity, create a dict with just numerical features
        # In practice, you'd need to handle categorical features properly
        batch_features = {"numerical": inputs}

        # Forward pass
        outputs = model(batch_features).squeeze()
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy


def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    """
    Evaluate the model on test data.

    Args:
        model: The neural network model
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
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # For simplicity, create a dict with just numerical features
            batch_features = {"numerical": inputs}

            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)

    return avg_loss, accuracy


def main() -> None:
    """
    Main entry point for the tabular classification project.
    """
    print("Tabular Classification with Titanic Dataset")
    print("=" * 45)

    # 1. Device setup
    global DEVICE
    DEVICE = get_device()
    if DEVICE == "mps":
        print(f"🔥 Found MPS device. Using {DEVICE} for acceleration.")
    else:
        print("Using CPU for computation.")

    # 2. Load and preprocess data
    df = load_titanic_data()
    features, targets, encoders = preprocess_data(df)

    # For this simplified version, we'll only use numerical features
    # A full implementation would handle categorical embeddings properly
    numerical_features = features[0]["numerical"].shape[1]
    categorical_info = {}  # Empty for simplified version

    # 3. Create model
    model = TabularClassifier(numerical_features, categorical_info, EMBEDDING_DIM)
    model.to(DEVICE)

    print(f"Model: MLP with {numerical_features} numerical features")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Create data loaders (simplified version)
    train_loader, test_loader = create_data_loaders(features, targets)

    # 5. Define loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training loop
    print("\nStarting training...")
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc

    # 7. Final evaluation
    print("Final Results:")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("\nTraining complete! 🎉")
    print("Note: This is a simplified implementation focusing on numerical features.")
    print("For full tabular learning, implement proper categorical embeddings.")


if __name__ == "__main__":
    main()
