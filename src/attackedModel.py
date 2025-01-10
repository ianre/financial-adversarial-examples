import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import TimeSeries, CNN1DModel, evaluate_model, train_model
from sklearn.metrics import accuracy_score, matthews_corrcoef


def load_data(attacked=False):
    if attacked:
        attacked_path = os.path.join("data", "processed", "attacked_data_fgsm.csv")
        if not os.path.exists(attacked_path):
            raise FileNotFoundError("Adversarial data not found. Please generate it first.")
        df = pd.read_csv(attacked_path, index_col=0)
        prices = df['Close'].values
        labels = df['label'].values
    else:
        processed_path = os.path.join("data", "processed", "training_data.csv")
        if not os.path.exists(processed_path):
            raise FileNotFoundError("Please provide a valid CSV with 'Close' column.")

        df = pd.read_csv(processed_path, index_col=0)
        prices = df['Close'].values
        labels = (pd.Series(prices).shift(-1) > pd.Series(prices)).astype(int).values
        prices = prices[:-1]
        labels = labels[:-1]

    return prices, labels

def evaluate_model(attacked=False):
    '''
    Evaluate model on either the original or attacked dataset.
    Returns:
        all_labels: ground truth labels
        all_preds: predicted labels (0 or 1)
        avg_loss: average BCE loss on the dataset
    '''
    prices, labels = load_data(attacked=attacked)
    window_size = 28
    dataset = TimeSeries(prices, labels, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = CNN1DModel()
    model_path = os.path.join("experiments", "results", "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Please train the model first.")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = nn.BCELoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            preds = torch.argmax(out, dim=1)
            # one-hot encode y for loss calculation
            y_one_hot = torch.zeros((y.size(0), 2))
            for i, val in enumerate(y):
                y_one_hot[i, val.item()] = 1.0
            y_one_hot = y_one_hot.float()

            loss = criterion(out, y_one_hot)
            total_loss += loss.item()
            count += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / count if count > 0 else 0
    return all_labels, all_preds, avg_loss

def fgsm_attack(model, loss_fn, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)

    y_one_hot = torch.zeros((y.size(0), 2))
    for i, val in enumerate(y):
        y_one_hot[i, val.item()] = 1.0
    y_one_hot = y_one_hot.float()

    loss = loss_fn(output, y_one_hot)
    model.zero_grad()
    loss.backward()
    data_grad = x_adv.grad.data
    perturbed_data = x_adv + epsilon * data_grad.sign()
    return perturbed_data.detach()

def generate_adversarial_data(epsilon=4.54):
    prices, labels = load_data(attacked=False)
    window_size = 28
    dataset = TimeSeries(prices, labels, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_path = os.path.join("experiments", "results", "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Please train the model first.")

    model = CNN1DModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.BCELoss()

    perturbed_series = prices.copy().astype(float)
    model.requires_grad_(True)

    for i, (x, y) in enumerate(dataloader):
        x_adv = fgsm_attack(model, criterion, x, y, epsilon)
        perturbed_window = x_adv.squeeze(0).squeeze(0).cpu().numpy()
        start_idx = i
        end_idx = i + window_size
        prev_window = perturbed_series[start_idx:end_idx]
        perturbed_series[start_idx:end_idx] = perturbed_window

        plt.figure()
        plt.plot(range(start_idx, end_idx), perturbed_window, marker='o', label=f'Perturbed Window {i}')
        plt.plot(range(start_idx, end_idx), prev_window, marker='x', label=f'Orig. Window {i}')
        plt.title(f'Perturbed Window Indices {start_idx}-{end_idx - 1}')
        plt.xlabel('Index')
        plt.ylabel('Price/Value')
        plt.legend()
        plt.tight_layout()
        plt.show()

    adv_labels = (pd.Series(perturbed_series).shift(-1) > pd.Series(perturbed_series)).astype(int).values[:-1]
    perturbed_series = perturbed_series[:-1]

    adv_df = pd.DataFrame({"Close": perturbed_series, "label": adv_labels})
    attacked_path = os.path.join("data", "processed", "attacked_data_fgsm.csv")
    os.makedirs(os.path.dirname(attacked_path), exist_ok=True)
    adv_df.to_csv(attacked_path)
    print(f"Adversarial data saved to {attacked_path}")


def attack_range():
    """
    Sweep epsilon values from 0 to 5 in increments of 0.1.
    """
    from sklearn.metrics import accuracy_score
    epsilon_steps = np.arange(1, 50.1, 1.0)
    accuracies = []
    losses = []
    for eps in epsilon_steps:
        print(f"\nGenerating adversarial data with epsilon={eps:.2f} ...")
        generate_adversarial_data(epsilon=eps)
        labels, preds, avg_loss = evaluate_model(attacked=True)
        acc = accuracy_score(labels, preds)
        accuracies.append(acc)
        losses.append(avg_loss)
        print(f"Epsilon={eps:.2f}, Loss={avg_loss:.4f}, Accuracy={acc:.4f}")
    # Optionally, plot epsilon vs. accuracy
    plt.figure(figsize=(8,5))
    plt.plot(epsilon_steps, accuracies, marker='o', label='Accuracy')
    plt.title("Accuracy vs. Epsilon (FGSM Attack Range)")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_loss():
    original_labels, original_preds, original_loss = evaluate_model(attacked=False)
    attacked_labels, attacked_preds, attacked_loss = evaluate_model(attacked=True)

    print(f"Original model Loss: {original_loss:.4f}")
    print(f"Attacked model Loss: {attacked_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Evaluate, Attack, or Compare Model Predictions")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "attack", "attack_range", "compare"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "evaluate":
        labels, preds, loss_val = evaluate_model(attacked=False)
        print(f"Original Data Loss: {loss_val}")
    elif args.mode == "attack":
        generate_adversarial_data(epsilon=100)
    elif args.mode == "attack_range":
        attack_range()
    elif args.mode == "compare":
        compare_loss()