import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from lag_llama.model import TimeSeries, CNN1DModel, CNN1DModel2, evaluate_model, train_model
from types import ModuleType


loss_func = nn.CrossEntropyLoss()

def load_data(attacked=False, type='fgsm'):
    if attacked:
        attacked_path = os.path.join("data", "processed", f"attacked_data_{type}.csv")
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

def load_data_llama(attacked=False, type='fgsm'):
    if attacked:
        attacked_path = os.path.join("data", "processed", f"attacked_data_{type}_llama.csv")
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

def evaluate_model(attacked=False, type='fgsm'):
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



def generate_adversarial_data(epsilon=4.54, type='fgsm'):
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

    idx = 0
    for i, (x, y) in enumerate(dataloader):
        idx+=1
        if type == 'fgsm':

            x_adv = fgsm_attack(model, x, y, epsilon)
            '''
            print("# fgsm_attack input x:" + str(x.shape) + "  x_adv:" + str(x_adv.shape))   
            x1 = x.squeeze().detach().cpu().numpy()
            x_adv1 = x_adv.squeeze().detach().cpu().numpy()
            perturbation = x_adv1 - x1

            df = pd.DataFrame({
                "Timestep": list(range(len(x1))),
                "Original": x1,
                "Adversarial": x_adv1,
                "Perturbation": perturbation
            })
            print(df)
            '''
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

    attacked_path = os.path.join("data", "processed", f"attacked_data_{type}.csv")
    os.makedirs(os.path.dirname(attacked_path), exist_ok=True)
    adv_df.to_csv(attacked_path)
    print("generate_adversarial_data iters:" + str(idx))
    print(f"Adversarial data saved to {attacked_path}")

def generate_adversarial_data_cpu(epsilon=4.54, type='fgsm'):
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

    idx = 0
    for i, (x, y) in enumerate(dataloader):
        idx+=1
        if type == 'fgsm':

            x_adv = fgsm_attack(model, x, y, epsilon)
            '''
            print("# fgsm_attack input x:" + str(x.shape) + "  x_adv:" + str(x_adv.shape))   
            x1 = x.squeeze().detach().cpu().numpy()
            x_adv1 = x_adv.squeeze().detach().cpu().numpy()
            perturbation = x_adv1 - x1

            df = pd.DataFrame({
                "Timestep": list(range(len(x1))),
                "Original": x1,
                "Adversarial": x_adv1,
                "Perturbation": perturbation
            })
            print(df)
            '''
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

    attacked_path = os.path.join("data", "processed", f"attacked_data_{type}.csv")
    os.makedirs(os.path.dirname(attacked_path), exist_ok=True)
    adv_df.to_csv(attacked_path)
    print("generate_adversarial_data iters:" + str(idx))
    print(f"Adversarial data saved to {attacked_path}")

def generate_adversarial_llama_cpu(epsilon=4.54, type='fgsm'):
    prices, labels = load_data(attacked=False)
    # prices=array([ 217.83,  222.84,  225.85, ..., 1065.85, 1060.2 , 1055.95])
    # labels=array([1, 1, 1, ..., 0, 0, 0])
    
    window_size = 28
    dataset = TimeSeries(prices, labels, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # dataloader=<torch.utils.data.dataloader.DataLoader object at 0x00000231FB8D3970>


    perturbed_series = prices.copy().astype(float)
    # perturbed_series=array([ 217.83,  222.84,  225.85, ..., 1065.85, 1060.2 , 1055.95])

    idx = 0
    for i, (x, y) in enumerate(dataloader):
        idx+=1
        # i = 0
        # x = tensor([[[217.8300, 222.8400, 225.8500, 233.0600, 233.6800, 235.1100, 236.0500,          232.0500, 233.3600, 233.7900, 222.6800, 218.4400, 199.9300, 213.9600,          221.7400, 216.7200, 217.3500, 216.9600, 213.6200, 216.5500, 201.0900,          198.2200, 190.9700, 192.7400, 184.1400, 184.7200, 179.5600, 181.4900]]])
        # y = tensor([0])
        if type == 'fgsm':
            x_adv = x
            # x_adv = tensor([[[213.2900, 218.3000, 230.3900, 228.5200, 229.1400, 239.6500, 240.5900,          227.5100, 228.8200, 229.2500, 227.2200, 222.9800, 204.4700, 209.4200,          226.2800, 212.1800, 221.8900, 212.4200, 209.0800, 221.0900, 205.6300,          193.6800, 186.4300, 197.2800, 179.6000, 180.1800, 175.0200, 186.0300]]])
        elif type == 'bim':
            x_adv = x
        else:
            raise ValueError("type must be either 'fgsm' or 'bim'")
            
        perturbed_window = x_adv.detach().numpy() #. squeeze(0).squeeze(0).cpu().numpy()
        perturbed_series[i:i + window_size] = perturbed_window
        # perturbed_series = array([ 213.29000854,  218.30000305,  230.38999939, ..., 1065.85      ,       1060.2       , 1055.95      ])
        if idx % 100 == 0: print("llama progress " + str(idx))
    adv_labels = (pd.Series(perturbed_series).shift(-1) > pd.Series(perturbed_series)).astype(int).values[:-1]
    perturbed_series = perturbed_series[:-1]

    # Save attacked portion
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




def plot_epsilon_accuracy(attack_type='bim'):
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
    parser = argparse.ArgumentParser(description="Train, Evaluate, Attack, or Compare Model Predictions")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "attack", "compare", "plot", "epsilon_plot"], required=True)
    parser.add_argument("--type", type=str, choices=["fgsm", "bim"])
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "evaluate":
        labels, preds, loss_val = evaluate_model(attacked=False)
        print(f"Original Data Loss: {loss_val}")
    elif args.mode == "attack":
        generate_adversarial_data(epsilon=4.54, type=args.type)
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


if __name__ == "__main__":
    main()