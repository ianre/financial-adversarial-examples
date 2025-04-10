# financial-adversarial-examples
This repo aims to replicate the results in "Investigating machine learning attacks on financial time series models"

## Setup
- The 1D-CNN model is already setup
- The lag-llama model requires some setup    
    - To download the pre-trained example model, run `huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir src\lag_llama\lag-llama`

## Usage
- Setup python environment using `requirements.txt`
- Run `python src\fetchData.py` to fetch the data
- Run `python .\src\prepareData.py --file <data file in data\raw> --show` (eg. `python .\src\prepareData.py --file all_stocks_2006-01-01_to_2018-01-01.csv --show`)
- Run `python .src\model.py`

## Kaggle Setup
This repo downloads the DJIA 30 Stock Time Series dataset from kaggle. Here is how to set this up:
1. Create kaggle.json from your Kaggle account page.
2. Place it in ~/.kaggle/kaggle.json (or C:\\Users\\<Username>\\.kaggle\\kaggle.json)
3. run `pip install kaggle`

## Model
This uses a 1-Dimensional Convolutional Neural Network as a classifier.
- Input: "window" to determine the size of the convolution window
- Output: A vector of logits for each sample, one for each class `[logit_0, logit_1]`. The larger value determines the label

## Labels
- Label = 1: If the price at time t+1 is higher than the price at time t.
- Label = 0: If the price at time t+1 is lower than or equal to the price at time t.