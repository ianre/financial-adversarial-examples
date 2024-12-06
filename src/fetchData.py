import os
import subprocess

def fetch(output_dir):
    """
    Download DJIA 30 Stock Time Series
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"kaggle datasets download -d szrlee/stock-time-series-20050101-to-20171231 -p {output_dir} --unzip"
    subprocess.check_call(cmd, shell=True)

if __name__ == "__main__":
    raw_dir = os.path.join("data", "raw")
    fetch(raw_dir)
    print("data downloaded and unzipped to:", raw_dir)
