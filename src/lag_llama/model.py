import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self):
        super(CNN1DModel2, self).__init__()
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
        x = torch.rrelu(x)
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

    window_size = 28
    dataset = TimeSeries(prices, labels, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNN1DModel()
    optimizer = optim.SGD(model.parameters(), lr=1.71176e-5, momentum=0.081)
    criterion = nn.BCELoss()

    epochs = 10
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

    model_path = os.path.join("experiments", "results", "model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved at:", model_path)

def evaluate_model():
    prices, labels = load_data()
    window_size = 28
    dataset = TimeSeries(prices, labels, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = CNN1DModel()
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
