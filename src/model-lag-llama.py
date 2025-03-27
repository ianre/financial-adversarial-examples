import os
import sys
from itertools import islice
from re import escape
import argparse

import importlib.util
import torch
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
import yfinance as yf

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ll_path = os.path.join(base_dir, "lag-llama")

if os.path.isdir(ll_path):
    sys.path.append(ll_path)
    
    print(f"'{ll_path}' exists; added to path")
else:
    print(f"'{ll_path}' does not exist or is not a directory. Please clone lag-llama in base directory")


module_path = r"lag-llama\lag_llama\gluon\estimator.py"
spec = importlib.util.spec_from_file_location("estimator", module_path)
estimator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimator_module)
LagLlamaEstimator = estimator_module.LagLlamaEstimator

def load_data():
    attacked_path = os.path.join("data", "processed", "attacked_data.csv")
    if os.path.exists(attacked_path):
        df = pd.read_csv(attacked_path, index_col=0)
    else:
        processed_path = os.path.join("data", "processed", "cleaned_data.csv")
        df = pd.read_csv(processed_path, index_col=0)
    return df

def get_lag_llama_dataset(df):
    """
    Converts to GluonTS ListDataset.
    Assumes df.index contains timestamps and that the 'Close' column is the target series.
    """
    from gluonts.dataset.common import ListDataset
    try:
        start = pd.to_datetime(df.index[0]) # Try to convert the first index element to a timestamp
    except Exception:
        start = pd.Timestamp("2020-01-01") # default start date.
    return ListDataset([{"start": start, "target": df["Close"].values}], freq="D")



def train_model(pretrained=False):
    """
    skip training and use checkpoint if pretrained==True
    """
    df = load_data()
    dataset = get_lag_llama_dataset(df)

    prediction_length = 60
    num_samples = 1060

    if pretrained:
        print("Using pre-trained lag-llama - skipping training.")
        return None

    # Load the base checkpoint to extract hyperparameters
    ckpt_path = os.path.join("model", "lag-llama.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Pre-trained checkpoint not found at {ckpt_path}.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    # Instantiate the lag-llama estimator for training.
    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=prediction_length,
        context_length=32,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args.get("scaling", None),
        time_feat=estimator_args["time_feat"],
        batch_size=64,
        num_parallel_samples=num_samples,
        trainer_kwargs={"max_epochs": 50},
    )
    print("Training lag-llama model on training dataset...")
    predictor = estimator.train(
        dataset,
        cache_data=True,
        shuffle_buffer_length=1000,
    )
    # Save the trained predictor for later inference.
    trained_ckpt_path = os.path.join("experiments", "results", "lag_llama_trained.ckpt")
    os.makedirs(os.path.dirname(trained_ckpt_path), exist_ok=True)
    torch.save(predictor, trained_ckpt_path)
    print("Trained lag-llama model saved at:", trained_ckpt_path)
    return predictor

def evaluate_model():
    """
    Run inference using lag-llama on time series data and plot forecasts.
    Loads either the pre-trained or the newly trained model.
    """
    df = load_data()
    dataset = get_lag_llama_dataset(df)

    prediction_length = 60
    num_samples = 1060
    pretrained = True

    if pretrained:
        ckpt_path = os.path.join("model", "lag-llama.ckpt")
        print("Evaluating using pre-trained lag-llama model.")
    else:
        ckpt_path = os.path.join("experiments", "results", "lag_llama_trained.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Trained lag-llama model not found. Please train the model first.")
        print("Using new checkpoint")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Build predictor from checkpoint
    # reconstruct the estimator and create the predictor if the checkpoint contains hyper_parameters
    if isinstance(ckpt, dict) and "hyper_parameters" in ckpt:
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=prediction_length,
            context_length=32,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args.get("scaling", None),
            time_feat=estimator_args["time_feat"],
            batch_size=1,
            num_parallel_samples=num_samples,
        )
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)
    else:
        predictor = ckpt

    from gluonts.evaluation import make_evaluation_predictions
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    # Plot forecasts for the first whateevr series
    plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter("%b, %d")
    for idx, (forecast, ts) in enumerate(forecasts[:9]):
        ax = plt.subplot(3, 3, idx + 1)
        plt.plot(ts[(-4 * prediction_length):].to_timestamp(), label="Target")
        forecast.plot(color="g")
        plt.xticks(rotation=60)
        ax.xaxis.set_major_formatter(date_formater)
        plt.title(f"Forecast for series {forecast.item_id}")
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train or Evaluate the Lag-Llamma")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"])
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "evaluate":
        evaluate_model()
