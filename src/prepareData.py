import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(input_path, filename, output_path):
    csv_path = os.path.join(input_path, filename)
    df = pd.read_csv(csv_path)

    df.to_csv(output_path)
    return df

def show_data(df):
    print("Last few rows of the processed dataset:")
    print(df.tail(10))

    plt.figure(figsize=(10, 4))
    df['Close'].tail(9000).plot(title="Close Price (Normalized) - Last 200 Observations")
    plt.xlabel("Date")
    plt.ylabel("Normalized Close Price")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Financial Time Series Data")
    parser.add_argument("--file", type=str, required=True, help="Raw CSV filename located in data/raw directory.")
    parser.add_argument("--show", action="store_true", help="Show a glimpse of the final dataset and plot")
    args = parser.parse_args()

    raw_path = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "cleaned_data.csv")

    df = preprocess_data(raw_path, args.file, processed_path)
    print("Data preprocessed and saved to:", processed_path)

    if args.show:
        show_data(df)
