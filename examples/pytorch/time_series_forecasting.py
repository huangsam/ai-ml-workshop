"""
Time Series Forecasting with PyTorch

This script demonstrates time series forecasting using synthetic weather data and LSTM.
It covers sequential data fundamentals: RNNs, sequence processing, and temporal patterns.

Dataset: Synthetic weather data (temperature, humidity, wind speed)
Model: LSTM network for multi-step forecasting
"""

from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from utils import get_device

# --- 1. CONFIGURATION CONSTANTS ---
SEQUENCE_LENGTH = 24  # Look back 24 hours
PREDICTION_HORIZON = 6  # Predict next 6 hours
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64  # LSTM hidden units
NUM_LAYERS = 2  # LSTM layers
DEVICE = "cpu"  # Default to CPU; will check for MPS acceleration


class WeatherDataset(Dataset):
    """
    Custom Dataset for time series forecasting.
    """

    def __init__(self, data: np.ndarray, seq_length: int, pred_horizon: int):
        """
        Initialize dataset.

        Args:
            data: Normalized time series data
            seq_length: Number of past time steps to use for prediction
            pred_horizon: Number of future time steps to predict
        """
        self.data = data
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon

    def __len__(self) -> int:
        return len(self.data) - self.seq_length - self.pred_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: past seq_length time steps
        x = self.data[idx : idx + self.seq_length]
        # Target: next pred_horizon time steps
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.pred_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMForecaster(nn.Module):
    """
    LSTM-based time series forecasting model.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, pred_horizon: int):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Number of output features (should be pred_horizon * num_features)
            pred_horizon: Number of time steps to predict
        """
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.pred_horizon = pred_horizon

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Output predictions of shape (batch_size, pred_horizon, num_features)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        # Reshape to (batch_size, pred_horizon, num_features)
        # Assuming output_size = pred_horizon * num_features
        batch_size = x.size(0)
        num_features = self.output_size // self.pred_horizon  # This should be 3 for our case
        out = out.view(batch_size, self.pred_horizon, num_features)

        return out


def generate_weather_data(num_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic weather data for demonstration.

    Args:
        num_samples: Number of time steps to generate

    Returns:
        DataFrame with temperature, humidity, and wind speed
    """
    print("Generating synthetic weather data...")

    np.random.seed(42)

    # Generate time index (hourly)
    time_index = pd.date_range(start="2020-01-01", periods=num_samples, freq="H")

    # Temperature: seasonal pattern + daily variation + noise
    seasonal_temp = 20 + 10 * np.sin(2 * np.pi * np.arange(num_samples) / (24 * 365))  # Annual cycle
    daily_temp = 5 * np.sin(2 * np.pi * np.arange(num_samples) / 24)  # Daily cycle
    temp_noise = np.random.normal(0, 2, num_samples)
    temperature = seasonal_temp + daily_temp + temp_noise

    # Humidity: correlated with temperature + noise
    humidity = 60 + 0.5 * temperature + np.random.normal(0, 10, num_samples)
    humidity = np.clip(humidity, 0, 100)  # Bound between 0-100%

    # Wind speed: random with some autocorrelation
    wind_speed = np.random.exponential(5, num_samples)  # Exponential distribution
    wind_speed = np.clip(wind_speed, 0, 30)  # Reasonable bounds

    # Create DataFrame
    df = pd.DataFrame({"temperature": temperature, "humidity": humidity, "wind_speed": wind_speed}, index=time_index)

    print(f"Generated {len(df)} hours of weather data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Temperature range: {df['temperature'].min():.2f}°C to {df['temperature'].max():.2f}°C")
    print(f"Humidity range: {df['humidity'].min():.2f}% to {df['humidity'].max():.2f}%")
    print(f"Wind speed range: {df['wind_speed'].min():.2f} to {df['wind_speed'].max():.2f} m/s")
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, MinMaxScaler]:
    """
    Prepare data for training and testing.

    Args:
        df: Raw weather data DataFrame

    Returns:
        Tuple of (train_loader, test_loader, scaler)
    """
    print("Preparing data for training...")

    # Extract features
    features = df.values  # Shape: (num_samples, num_features)

    # Normalize data
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Split into train/test (80/20)
    train_size = int(0.8 * len(scaled_features))
    train_data = scaled_features[:train_size]
    test_data = scaled_features[train_size:]

    # Create datasets
    train_dataset = WeatherDataset(train_data, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    test_dataset = WeatherDataset(test_data, SEQUENCE_LENGTH, PREDICTION_HORIZON)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Input shape: ({SEQUENCE_LENGTH}, {features.shape[1]})")
    print(f"Output shape: ({PREDICTION_HORIZON}, {features.shape[1]})")

    return train_loader, test_loader, scaler


def train_model(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The LSTM model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: str) -> float:
    """
    Evaluate the model on test data.

    Args:
        model: The LSTM model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        Average test loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(test_loader)


def main() -> None:
    """
    Main entry point for the time series forecasting project.
    """
    print("Time Series Forecasting with LSTM")
    print("=" * 35)

    # 1. Device setup
    global DEVICE
    DEVICE = get_device()
    if DEVICE == "mps":
        print(f"🔥 Found MPS device. Using {DEVICE} for acceleration.")
    else:
        print("Using CPU for computation.")

    # 2. Generate and prepare data
    weather_df = generate_weather_data()
    train_loader, test_loader, scaler = prepare_data(weather_df)

    # 3. Create model
    input_size = weather_df.shape[1]  # Number of features (temp, humidity, wind)
    output_size = input_size * PREDICTION_HORIZON  # Predict all features for all horizons

    model = LSTMForecaster(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size, PREDICTION_HORIZON)
    model.to(DEVICE)

    print(f"Model: LSTM with {NUM_LAYERS} layers, {HIDDEN_SIZE} hidden units")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Define loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training loop
    print("\nStarting training...")
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, DEVICE)
        test_loss = evaluate_model(model, test_loader, criterion, DEVICE)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        if test_loss < best_loss:
            best_loss = test_loss
            # Could save model here: torch.save(model.state_dict(), 'best_model.pth')

    # 6. Final evaluation
    print("\nFinal Results:")
    print(f"Best Test Loss: {best_loss:.4f}")
    print("\nTraining complete! 🎉")
    print("Note: This is a basic LSTM implementation. For production use,")
    print("consider techniques like teacher forcing, attention mechanisms,")
    print("and proper validation with walk-forward validation.")


if __name__ == "__main__":
    main()
