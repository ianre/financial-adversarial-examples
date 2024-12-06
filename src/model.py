import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class TimeSeriesClassificationDataset(Dataset):
    def __init__(self, data, labels, window_size=30):
        self.data = data
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]
        y = self.labels[idx+self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class CNN1DModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, kernel_size=3, hidden_dim=64, window_size=30):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        conv_out_length = (window_size - (kernel_size - 1)) // 2
        self.fc = nn.Linear(conv_out_length * hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_data():
    attacked_path = os.path.join("data", "processed", "attacked_data.csv")
    if os.path.exists(attacked_path):
        df = pd.read_csv(attacked_path, index_col=0)
        prices = df['Close'].values
        labels = df['label'].values
    else:
        processed_path = os.path.join("data", "processed", "cleaned_data.csv")
        df = pd.read_csv(processed_path, index_col=0)
        prices = df['Close'].values
        labels = (pd.Series(prices).shift(-1) > pd.Series(prices)).astype(int).values
        prices = prices[:-1]
        labels = labels[:-1]

    return prices, labels

def train_model():
    prices, labels = load_data()

    window_size = 30
    dataset = TimeSeriesClassificationDataset(prices, labels, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNN1DModel(window_size=window_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    model_path = os.path.join("experiments", "results", "model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved at:", model_path)

def evaluate_model():
    prices, labels = load_data()
    window_size = 30
    dataset = TimeSeriesClassificationDataset(prices, labels, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = CNN1DModel(window_size=window_size)
    model_path = os.path.join("experiments", "results", "model.pth")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train or Evaluate the 1D CNN Model")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"])
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "evaluate":
        evaluate_model()
