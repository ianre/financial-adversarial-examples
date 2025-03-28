import os
import sys
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch.utils.data import Dataset, DataLoader
from model import TimeSeries, CNN1DModel, train_model
from pathlib import Path
import numpy as np
from types import ModuleType

from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.evaluation import make_evaluation_predictions


loss_func = nn.CrossEntropyLoss()

def load_data(attacked=False, type='fgsm', model_type="1dcnn"):
    if attacked:
        attacked_path = os.path.join("data", "processed", f"attacked_data_{type}_{model_type}.csv")
        if not os.path.exists(attacked_path):
            raise FileNotFoundError("Adversarial data not found. Please generate it first.")
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

def evaluate_model(model_type="1dcnn", attacked=False, type='fgsm'):
    if model_type == "1dcnn":
        return evaluate_model_1dcnn(attacked, type)
    elif model_type =="llama":
        return evaluate_model_llama(attacked, type)

def evaluate_model_1dcnn(attacked=False, type='fgsm'):
    prices, labels = load_data(attacked, type)
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
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)  # out shape: (batch_size, 2), raw logits
            preds = torch.argmax(out, dim=1)  # Get predicted class index (0 or 1)

            # Compute loss
            loss = loss_func(out, y.long())
            total_loss += loss.item()
            count += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / count if count > 0 else 0
    return all_labels, all_preds, avg_loss

def get_window_dataset(window):
    """
    Given a 1D array 'window', creates a GluonTS ListDataset for a single time series.
    """
    df_window = pd.DataFrame({"Close": window})
    # Create a dummy datetime index; here we assume daily frequency
    df_window.index = pd.date_range(start="2020-01-01", periods=len(df_window), freq="D")
    from gluonts.dataset.common import ListDataset
    return ListDataset([{"start": df_window.index[0], "target": df_window["Close"].values}], freq="D")

def build_llama_predictor(prediction_length, context_length, num_samples, ckpt_path):
    """
    Loads the lag-llama checkpoint and builds a predictor.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    # We assume the checkpoint contains hyperparameters under "hyper_parameters"
    if isinstance(ckpt, dict) and "hyper_parameters" in ckpt:
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=prediction_length,
            context_length=context_length,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            batch_size=1,
            num_parallel_samples=num_samples,
        )
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)
    else:
        predictor = ckpt  # assume checkpoint is directly a predictor
        lightning_module = ""
    return predictor, lightning_module

def evaluate_model_llama(attacked=False, type='fgsm'):
    # Load financial data (prices and labels)
    prices, labels = load_data(attacked, type)
    
    window_size = 28         # size of the input window (context)
    prediction_length = 1    # predicting 1 step ahead
    num_samples = 100        # number of samples from forecast distribution

    ckpt_path = getLlamaModel()

    # Build a predictor with context_length equal to the window_size
    predictor, lightning_module = build_llama_predictor(prediction_length, context_length=window_size, num_samples=num_samples, ckpt_path=ckpt_path)
    
    all_true_labels = []
    all_pred_labels = []
    
    # Loop over sliding windows of the price series
    for i in range(len(prices) - window_size):
        window = prices[i:i+window_size]
        true_label = labels[i]  # label corresponds to whether prices[i+window_size] > prices[i+window_size-1]
        
        # Create a GluonTS dataset from the current window
        window_dataset = get_window_dataset(window)
        
        # Generate forecast for the next step using Lag-Llama
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=window_dataset,
            predictor=predictor,
            num_samples=num_samples
        )
        # Since there's one time series, get the first forecast object
        forecast = list(forecast_it)[0]
        # For prediction_length=1, the point forecast can be taken as the mean of the distribution
        pred_price = forecast.mean[0]
        # Derive predicted label: 1 if forecasted next price > last observed price, else 0
        pred_label = 1 if pred_price > window[-1] else 0
        
        all_true_labels.append(true_label)
        all_pred_labels.append(pred_label)
    
    # Compute average loss as the misclassification rate (or 1 - accuracy)
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    accuracy = np.mean(all_true_labels == all_pred_labels)
    avg_loss = 1 - accuracy
    
    # Optionally, print or plot a few examples (omitted for brevity)
    print("Accuracy:", accuracy)
    print("Average Loss (Misclassification Rate):", avg_loss)
    
    return all_true_labels, all_pred_labels, avg_loss

def fgsm_attack(model, x, y, epsilon=4.54):
    x_adv = x.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(x_adv)

    # Calculate ce loss
    loss = loss_func(output, y)

    # Zero existing grads
    model.zero_grad()

    # Compute grads of loss wrt x_adv
    loss.backward()

    # Get element wise sign of the data gradient
    data_grad = x_adv.grad.data.sign()

    # Apply perturbation
    x_adv = x_adv + epsilon * data_grad

    return x_adv.detach()

def basic_iterative_method(model, x, y, epsilon=4.54, num_iterations=50):

    # per-iteration step size
    step_size = epsilon / num_iterations
    x_adv = x.clone().detach()

    for i in range(num_iterations):
        x_adv.requires_grad_()
        output = model(x_adv)
        loss = loss_func(output, y)

        model.zero_grad()
        loss.backward()

        # Update adversarial series using step size
        x_adv = x_adv + step_size * x_adv.grad.sign()
        
        x_adv = x_adv.detach()

    return x_adv

def generate_adversarial_data(model_type="1dcnn", epsilon=4.54, type='fgsm'):
    if model_type == "1dcnn":
        generate_adversarial_data_cnn(epsilon, type)
    elif model_type == "llama":
        generate_adversarial_data_llama(epsilon, type)

def generate_adversarial_data_cnn(epsilon=4.54, type='fgsm'):
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

    perturbed_series = prices.copy().astype(float)

    for i, (x, y) in enumerate(dataloader):
        if type == 'fgsm':
            x_adv = fgsm_attack(model, x, y, epsilon)
        elif type == 'bim':
            x_adv = basic_iterative_method(model, x, y, epsilon, 20)
        else:
            raise ValueError("type must be either 'fgsm' or 'bim'")
            
        perturbed_window = x_adv.squeeze(0).squeeze(0).cpu().numpy()
        perturbed_series[i:i + window_size] = perturbed_window

    adv_labels = (pd.Series(perturbed_series).shift(-1) > pd.Series(perturbed_series)).astype(int).values[:-1]
    perturbed_series = perturbed_series[:-1]

    # Save attacked portion
    adv_df = pd.DataFrame({
        "Close": perturbed_series,
        "label": adv_labels
    })

    attacked_path = os.path.join("data", "processed", f"attacked_data_{type}_1dcnn.csv")
    os.makedirs(os.path.dirname(attacked_path), exist_ok=True)
    adv_df.to_csv(attacked_path)
    print(f"Adversarial data saved to {attacked_path}")
    
def generate_adversarial_data_llama(epsilon=4.54, type='fgsm'):

    prices, labels = load_data(attacked=False)
    window_size = 28
    perturbed_series = prices.copy().astype(float)

    ckpt_path = getLlamaModel()
    prediction_length = 1
    num_samples = 100
    context_length = window_size
    
    predictor, lightning_module = build_llama_predictor(prediction_length, context_length=window_size, num_samples=num_samples, ckpt_path=ckpt_path)

    for i in range(len(prices) - window_size):
        # Prepare the input window: for lag-llama, we assume a tensor shape [batch, context_length, feature_dim]
        x = torch.tensor(prices[i:i+window_size], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # shape: [1, window_size, 1]
        # Use the corresponding label as a dummy target (not used in gradient calculation here)
        y = torch.tensor([labels[i]], dtype=torch.float32)
        
        # Compute adversarial perturbation using FGSM or BIM
        if type == 'fgsm':
            x_adv = fgsm_attack(lightning_module, x, y, epsilon)
        elif type == 'bim':
            x_adv = basic_iterative_method(lightning_module, x, y, epsilon, 20)
        else:
            raise ValueError("type must be either 'fgsm' or 'bim'")
        
        # Assume x_adv has shape [1, window_size, 1]; squeeze to get a 1D array
        perturbed_window = x_adv.squeeze().cpu().numpy()
        # Replace the original window in the perturbed series
        perturbed_series[i:i + window_size] = perturbed_window

    adv_labels = (pd.Series(perturbed_series).shift(-1) > pd.Series(perturbed_series)).astype(int).values[:-1]
    perturbed_series = perturbed_series[:-1]
    
    # Save the adversarial data as a CSV file
    adv_df = pd.DataFrame({
        "Close": perturbed_series,
        "label": adv_labels
    })
    attacked_path = os.path.join("data", "processed", f"attacked_data_{type}_llama.csv")
    os.makedirs(os.path.dirname(attacked_path), exist_ok=True)
    adv_df.to_csv(attacked_path)
    print(f"Adversarial data saved to {attacked_path}")


def compare_predictions(attack_type):
    """
    Evaluate the model on both original and adversarial data,
    then compare and print the loss, accuracy, and prediction change rate.
    """
    original_labels, original_preds, original_loss = evaluate_model(attacked=False)
    attacked_labels, attacked_preds, attacked_loss = evaluate_model(attacked=True, type=attack_type)

    # Calculate prediction changes
    prediction_changes = sum(1 for x, y in zip(original_preds, attacked_preds) if x != y)
    change_rate = prediction_changes / len(original_preds) if original_preds else 0

    # Calculate accuracies
    original_accuracy = sum(1 for x, y in zip(original_labels, original_preds) if x == y) / len(original_labels)
    attacked_accuracy = sum(1 for x, y in zip(attacked_labels, attacked_preds) if x == y) / len(attacked_labels)

    print(f"Original model - Loss: {original_loss:.4f}, Accuracy: {original_accuracy * 100:.2f}%")
    print(f"Attacked model - Loss: {attacked_loss:.4f}, Accuracy: {attacked_accuracy * 100:.2f}%")
    print(f"Number of predictions changed: {prediction_changes} ({change_rate * 100:.2f}%)")
    
    return original_accuracy, attacked_accuracy

def plot_attacked_vs_original_data(attack_type='fgsm'):
    """
    Plot the original and attacked data
    """
    # Load original and attacked data
    original_prices, _ = load_data(attacked=False)
    original_prices = original_prices[2194:2222]
    attacked_prices, _ = load_data(attacked=True, type=attack_type)
    #attacked_prices = attacked_prices[2194:2222]
    print(len(original_prices))
    print(len(attacked_prices))
    
    # Shift both series so that the starting original price becomes 0
    offset = original_prices[0]
    original_prices_shifted = original_prices - offset
    attacked_prices_shifted = attacked_prices - offset

    # Create the plot using the shifted data
    plt.figure(figsize=(25, 7))
    plt.plot(original_prices_shifted, label='Original', alpha=1)
    plt.plot(attacked_prices_shifted, label='Adversarial', alpha=1)
    
    plt.title(f'Original vs Adversarial Time Series ({attack_type.upper()} Attack) [Shifted]')
    plt.xlabel('Days')
    plt.ylabel('Price Change')
    plt.legend()
    plt.grid(True, alpha=0.4)

    # Save the plot
    plot_path = os.path.join("experiments", "results", f"adversarial_comparison_{attack_type}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def plot_epsilon_accuracy(model_type="1dcnn", attack_type='bim'):
    """
    Evaluate the model for a range of epsilon values under the specified attack,
    plot the model accuracy versus epsilon, and save the plot.
    """
    # Define epsilon values (including more points between 2 and 8)
    epsilons = [0.5, 1.0, 2.0, 3.0, 4.0, 4.54, 5.0, 6.0, 7.0, 8.0, 16.0, 17.0, 18.0]
    accuracies = []
    
    # Get original accuracy for reference
    original_labels, original_preds, _ = evaluate_model(attacked=False)
    original_accuracy = sum(1 for x, y in zip(original_labels, original_preds) if x == y) / len(original_labels)
    
    # Test each epsilon value
    for eps in epsilons:
        print(f"Testing epsilon: {eps}")
        generate_adversarial_data(epsilon=eps, type=attack_type)
        
        attacked_labels, attacked_preds, _ = evaluate_model(attacked=True, type=attack_type)
        accuracy = sum(1 for x, y in zip(attacked_labels, attacked_preds) if x == y) / len(attacked_labels)
        accuracies.append(accuracy)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    plt.plot(epsilons, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8,
             color='blue', label=f'{attack_type.upper()} Attack Accuracy')
    plt.axhline(y=original_accuracy, color='r', linestyle='--', label='Original Accuracy')
    
    plt.title(f'Model Accuracy vs Epsilon ({attack_type.upper()} Attack)')
    plt.xlabel('Epsilon Value')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Annotate each point with its accuracy value
    for eps, acc in zip(epsilons, accuracies):
        plt.annotate(f'{acc:.3f}', (eps, acc), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Save the plot
    plot_path = os.path.join("experiments", "results", f"epsilon_accuracy_{attack_type}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {plot_path}")

def main():    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    parser = argparse.ArgumentParser(description="Train, Evaluate, Attack, or Compare Model Predictions")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "attack", "compare", "plot", "epsilon_plot"], required=True)
    parser.add_argument("--type", type=str, choices=["fgsm", "bim"])
    parser.add_argument("--model", default="1dcnn", type=str, choices=["1dcnn", "llama"])
    args = parser.parse_args()

    if args.mode == "train":
        #train_model()
        print("implement train")
    elif args.mode == "evaluate":
        labels, preds, loss_val = evaluate_model(model_type=args.model,attacked=False)
        saveRun([loss_val],args.model,"single_evaluation")
        print(f"Original Data Loss: {loss_val}")
    elif args.mode == "attack":
        generate_adversarial_data(model_type=args.model, epsilon=4.54, type=args.type)
    elif args.mode == "compare":
        compare_predictions(args.type)
    elif args.mode == "plot":
        if not args.type:
            print("Please specify attack type with --type [fgsm/bim]")
        else:
            plot_attacked_vs_original_data(args.type)
    elif args.mode == "epsilon_plot":
        if not args.type:
            print("Please specify attack type with --type [fgsm/bim]")
        else:
            plot_epsilon_accuracy(args.type)

def saveRun(dataframe, model_type="1dcnn",eval_type="single_evaluation"):
    current_time = datetime.now().isoformat()
    evaluate_file_path = os.path.join("eval", eval_type, f"{model_type}.log")
    os.makedirs(os.path.dirname(evaluate_file_path), exist_ok=True)
    with open(evaluate_file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        dataframe.insert(0,current_time)
        writer.writerow(dataframe)

def getLlamaModel():
    ckpt_path = r"src\lag_llama\lag-llama\lag-llama.ckpt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("No lag-llama checkpoint found. Please train or download the model first.")        
    print("Using official pretrained Lag-Llama model.")
    return ckpt_path

if __name__ == "__main__":
    main()