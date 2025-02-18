import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(input_path, filename, output_path):
    csv_path = os.path.join(input_path, filename)
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    train_df = df[(df['Date'] >= '2006-01-01') & (df['Date'] <= '2014-12-31')]
    eval_df = df[(df['Date'] >= '2015-01-01') & (df['Date'] <= '2018-01-01')]

    train_path = os.path.join(os.path.dirname(output_path), "training_data.csv")
    eval_path = os.path.join(os.path.dirname(output_path), "evaluation_data.csv")

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    return train_df, eval_df

def preprocess_data_expand(input_path, filename, output_path, window_size):
    csv_path = os.path.join(input_path, filename)
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    
    df = df[['Date', 'Close']]
    
    # split data
    train_df = df[(df['Date'] >= '2006-01-01') & (df['Date'] <= '2014-12-31')]
    eval_df  = df[(df['Date'] >= '2015-01-01') & (df['Date'] <= '2018-01-01')]
    
    # Save only close prices
    train_close_df = train_df[['Close']]
    eval_close_df  = eval_df[['Close']]
    
    train_close_path = os.path.join(os.path.dirname(output_path), "training_data_only_close.csv")
    eval_close_path  = os.path.join(os.path.dirname(output_path), "evaluation_data_only_close.csv")
    
    train_close_df.to_csv(train_close_path, index=False)
    eval_close_df.to_csv(eval_close_path, index=False)
    
    def create_long_windows(df, window_size):
        values = df['Close'].values
        rows = []
        # For each window starting index:
        for i in range(len(values) - window_size + 1):
            window = values[i:i+window_size]
            # Normalize
            normalized_window = window - window[0]
            window_start_date = df.iloc[i]['Date']
            for day_idx, norm_val in enumerate(normalized_window, start=1):
                row = {
                    "Window_Start": window_start_date,
                    "Day_Index": day_idx,
                    "Normalized_Close": norm_val
                }
                rows.append(row)
        return pd.DataFrame(rows)

    train_long_windows = create_long_windows(train_df, window_size)
    eval_long_windows  = create_long_windows(eval_df, window_size)

    train_windows_path = os.path.join(os.path.dirname(output_path), "training_windows_only_close_long.csv")
    eval_windows_path  = os.path.join(os.path.dirname(output_path), "evaluation_windows_only_close_long.csv")
    
    train_long_windows.to_csv(train_windows_path, index=False)
    eval_long_windows.to_csv(eval_windows_path, index=False)
    
    return train_df, eval_df, train_long_windows, eval_long_windows


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
    parser.add_argument("--expand", action="store_true", help="save expand")
    args = parser.parse_args()

    raw_path = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    if(args.expand):

        processed_path = os.path.join(processed_dir, "cleaned_data.csv")

        train_df, eval_df, train_windows, eval_windows = preprocess_data_expand(raw_path, args.file, processed_path, 30)
        print("Data preprocessed and saved to:", processed_path)
    
    else:
        processed_path = os.path.join(processed_dir, "cleaned_data.csv")

        train_df, eval_df = preprocess_data(raw_path, args.file, processed_path)
        print("Data preprocessed and saved to:", processed_path)

        if args.show:
        #    show_data(df)
            show_data(train_df)
            show_data(eval_df)
