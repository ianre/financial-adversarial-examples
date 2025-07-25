import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler


#financial_adversarial_examples location
examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class TimeSeries(Dataset):
    def __init__(self, data, labels, window_size=30):
        self.data = data
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]
        y = self.labels[idx+self.window_size]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y, dtype=torch.long)

class TimeSeriesRegression(Dataset):
    def __init__(self, data, targets, window_size=30, pred_horizon=1):
        self.data = data
        self.targets = targets
        self.window_size = window_size
        self.pred_horizon = pred_horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.pred_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]
        # Get multiple target days
        y = self.targets[idx+self.window_size-1:idx+self.window_size-1+self.pred_horizon]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y, dtype=torch.float32)

class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(84, 250)
        self.fc2 = nn.Linear(250, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class CNN1DModel2(nn.Module):
    def __init__(self, pred_horizon=14):
        super(CNN1DModel2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(84, 250)
        self.fc2 = nn.Linear(250, pred_horizon)
        
        # Add batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def load_data():
    
    # Construct paths relative to examples directory
    attacked_path = os.path.join(examples_dir, "data", "processed", "attacked_data.csv")
    processed_path = os.path.join(examples_dir, "data", "processed", "cleaned_data.csv")
    
    # Check if data directory exists
    data_dir = os.path.join(examples_dir, "data", "processed")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found at {data_dir}. "
            "Please ensure you're running from the examples directory and data is properly set up."
        )
    
    if os.path.exists(attacked_path):
        df = pd.read_csv(attacked_path, index_col=0)
        prices = df['Close'].values
        labels = df['label'].values
    elif os.path.exists(processed_path):
        df = pd.read_csv(processed_path, index_col=0)
        prices = df['Close'].values
        labels = (pd.Series(prices).shift(-1) > pd.Series(prices)).astype(int).values
        prices = prices[:-1]
        labels = labels[:-1]
    else:
        raise FileNotFoundError(
            f"Neither attacked data ({attacked_path}) nor processed data ({processed_path}) found. "
            "Please ensure at least one of these files exists in the examples/data/processed directory."
        )

    return prices, labels

def load_regression_data(data_type='train'):
    """
    Load regression data from either training or evaluation dataset with normalization
    
    Args:
        data_type: str, either 'train' or 'eval' to specify which dataset to load
    """
    if data_type == 'train':
        data_path = os.path.join(examples_dir, "data", "processed", "GOOGL_train_univariate_regression_data.csv")
    else:
        data_path = os.path.join(examples_dir, "data", "processed", "GOOGL_test_univariate_regression_data.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
        
    # Load only the 'Close' column
    df = pd.read_csv(data_path)
    if 'Close' not in df.columns:
        raise ValueError(f"'Close' column not found in {data_path}")
    
    prices = df['Close'].values.reshape(-1, 1)
    
    # Initialize and fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_normalized = scaler.fit_transform(prices).flatten()
    
    # Target is the next day's normalized price
    targets = prices_normalized[1:]
    prices = prices_normalized[:-1]
    
    return prices, targets, scaler  # Return scaler for denormalization later

def train_model():
    prices, labels = load_data()

    window_size = 28
    dataset = TimeSeries(prices, labels, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNN1DModel2()
    optimizer = optim.SGD(model.parameters(), lr=1.71176e-5, momentum=0.081)
    criterion = nn.BCELoss()

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in dataloader:
            y_one_hot = torch.zeros((y.size(0), 2))
            for i, val in enumerate(y):
                if val.item() == 1:
                    y_one_hot[i,1] = 1.0
                else:
                    y_one_hot[i,0] = 1.0
            y_one_hot = y_one_hot.float()

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y_one_hot)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")


    model_path = os.path.join(examples_dir, "experiments", "models", "model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved at:", model_path)

def train_regression_model(window_size=28, batch_size=32, epochs=50, learning_rate=1e-4, pred_horizon=14):
    """
    Train the CNN1DModel2 for regression with normalized data
    """
    # Load and prepare normalized data
    prices, targets, scaler = load_regression_data(data_type='train')
    
    # Create dataset and dataloader
    dataset = TimeSeriesRegression(prices, targets, window_size=window_size, pred_horizon=pred_horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modify the last layer for regression
    model = CNN1DModel2(pred_horizon=pred_horizon)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            # Convert normalized loss to actual price scale for interpretability
            price_range = scaler.data_max_ - scaler.data_min_
            scaled_rmse = np.sqrt(avg_loss) * price_range[0]
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Average Loss (MSE): {avg_loss:.4f}")
            print(f"RMSE in price scale: ${scaled_rmse:.2f}")
            print("-" * 40)
    
    # Save model and scaler
    model_path = os.path.join(examples_dir, "experiments", "models", f"regression_model_{pred_horizon}day.pth")
    scaler_path = os.path.join(examples_dir, "experiments", "models", f"scaler_{pred_horizon}day.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state': {
            'data_min_': scaler.data_min_,
            'data_max_': scaler.data_max_,
            'scale_': scaler.scale_,
        }
    }, model_path)
    
    return model, scaler

def evaluate_model():
    prices, labels = load_data()
    window_size = 28
    dataset = TimeSeries(prices, labels, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = CNN1DModel()
    model_path = os.path.join(examples_dir, "experiments", "models", "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Please train the model first.")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = pd.Series(all_preds)
    all_labels = pd.Series(all_labels)

    plt.figure(figsize=(10,4))
    plt.plot(all_labels.values, label='Actual Labels', alpha=0.7)
    plt.plot(all_preds.values, label='Predicted Labels', alpha=0.7)
    plt.title("Model Predictions vs Actual Labels")
    plt.xlabel("Sample Index")
    plt.ylabel("Label")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_regression_model(window_size=28, batch_size=32, pred_horizon=14):
    """
    Evaluate the regression model with normalized data
    """
    # Load data and get ticker name from the CSV
    eval_df = pd.read_csv(os.path.join(examples_dir, "data", "processed", "evaluation_data.csv"))
    ticker_name = eval_df['Name'].iloc[0] if 'Name' in eval_df.columns else "Unknown"
    
    # Convert dates to datetime
    dates = pd.to_datetime(eval_df['Date'])
    
    prices, targets, scaler = load_regression_data(data_type='eval')
    dataset = TimeSeriesRegression(prices, targets, window_size=window_size, pred_horizon=pred_horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = CNN1DModel2(pred_horizon=pred_horizon)
    model_path = os.path.join(examples_dir, "experiments", "models", f"regression_model_{pred_horizon}day.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Please train the model first.")
    
    # Load model and scaler state
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            predictions.append(out.numpy())
            actuals.append(y.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # Denormalize predictions and actuals
    predictions_reshaped = predictions.reshape(-1, 1)
    actuals_reshaped = actuals.reshape(-1, 1)
    
    predictions_denorm = scaler.inverse_transform(predictions_reshaped).reshape(predictions.shape)
    actuals_denorm = scaler.inverse_transform(actuals_reshaped).reshape(actuals.shape)
    
    # Create continuous prediction line
    continuous_pred = []
    continuous_actual = []
    prediction_dates = []  # Store corresponding dates
    
    for i in range(0, len(predictions_denorm), pred_horizon):
        if i + pred_horizon <= len(predictions_denorm):
            continuous_pred.extend(predictions_denorm[i])
            continuous_actual.extend(actuals_denorm[i])
            # Add dates for this prediction window
            prediction_dates.extend(dates[i:i + pred_horizon])
    
    # Convert to pandas Series with DateTimeIndex for better plotting
    continuous_pred_series = pd.Series(continuous_pred, index=prediction_dates)
    continuous_actual_series = pd.Series(continuous_actual, index=prediction_dates)
    
    # Plot results with denormalized values and date x-axis
    plt.figure(figsize=(15, 6))
    plt.plot(continuous_actual_series.index, continuous_actual_series.values, 
             label='Actual Prices', alpha=0.7)
    plt.plot(continuous_pred_series.index, continuous_pred_series.values, 
             label='Predicted Prices', alpha=0.7)
    
    plt.title(f"1D CNN Prediction of {ticker_name} Price ({pred_horizon}-day horizon)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    
    # Add vertical lines at prediction window boundaries
    for i in range(pred_horizon, len(continuous_pred), pred_horizon):
        plt.axvline(x=prediction_dates[i], color='gray', linestyle='--', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Calculate metrics on denormalized values
    mse = np.mean((np.array(continuous_pred) - np.array(continuous_actual)) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(continuous_pred) - np.array(continuous_actual)))
    
    print(f"\nOverall metrics:")
    print(f"MSE: ${mse:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    
    plt.tight_layout()  # Adjust layout to prevent date labels from being cut off
    
    # Save the plot
    plot_path = os.path.join(examples_dir, "experiments", "results", f"{ticker_name}_prediction_regression.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')  # bbox_inches='tight' to prevent label cutoff
    plt.show()
    
    print(f"\nPlot saved to: {plot_path}")
    
    return continuous_pred_series, continuous_actual_series

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Evaluate the 1D CNN Model")
    parser.add_argument("--mode", type=str, 
                      choices=["train", "evaluate", "train_regression", "evaluate_regression"],
                      help="Mode of operation: train/evaluate classification or regression model")
    parser.add_argument("--window_size", type=int, default=28,
                      help="Size of the sliding window for time series")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training/evaluation")
    parser.add_argument("--epochs", type=int, default=300,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate for training")
    parser.add_argument("--pred_horizon", type=int, default=14,
                      help="Number of days to predict into the future")
    
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "evaluate":
        evaluate_model()
    elif args.mode == "train_regression":
        train_regression_model(
            window_size=args.window_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            pred_horizon=args.pred_horizon
        )
    elif args.mode == "evaluate_regression":
        evaluate_regression_model(
            window_size=args.window_size,
            batch_size=args.batch_size,
            pred_horizon=args.pred_horizon
        )
