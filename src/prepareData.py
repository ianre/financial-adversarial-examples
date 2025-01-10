import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(input_path, filename, output_path):
    csv_path = os.path.join(input_path, filename)
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # Split training (2006-01-01 to 2014-12-31) and evaluation (2015-01-01 to 2018-01-01)
    train_df = df[(df['Date'] >= '2006-01-01') & (df['Date'] <= '2014-12-31')]
    eval_df = df[(df['Date'] >= '2015-01-01') & (df['Date'] <= '2018-01-01')]

    train_path = os.path.join(os.path.dirname(output_path), "training_data.csv")
    eval_path = os.path.join(os.path.dirname(output_path), "evaluation_data.csv")

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    return train_df, eval_df

def show_data(df):
    print("Last few rows of the processed dataset:")
    try:
        print(df.tail(10))
    except:
        print("not a frame")
    obs = 9000

    plt.figure(figsize=(10, 4))
    df['Close'].tail(obs).plot(title="close Price (Normalized) - Last " + str(obs) + " Observations")
    plt.xlabel("Date")
    plt.ylabel("Normalized Close Price")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Financial Time Series Data")
    parser.add_argument("--file", type=str, required=True, help="Raw CSV filename located in data/raw directory.")
    parser.add_argument("--show", action="store_true", help="plot dataset")
    args = parser.parse_args()

    raw_path = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "cleaned_data.csv")

    train_df, eval_df = preprocess_data(raw_path, args.file, processed_path)
    print("Data preprocessed and saved to:", processed_path)

if args.show:
  #    show_data(df)
  show_data(train_df)
  show_data(eval_df)
