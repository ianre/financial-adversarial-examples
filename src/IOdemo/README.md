# Financial Adversarial Examples Demo

This demo provides a graphical user interface for experimenting with adversarial attacks on financial time series data. It allows you to:

1. Select a stock from available data
2. View the original time series
3. Apply FGSM or BIM attacks with adjustable parameters
4. Compare the original and perturbed time series
5. Evaluate model performance on both original and adversarial data

## Installation

1. Make sure you have Python 3.8 or later installed
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the demo from the root:
   ```bash
   python -m src.IOdemo.demo
   ```

2. In the GUI:
   - Select a stock from the dropdown menu
   - Adjust the epsilon parameter (controls attack strength)
   - Click "Apply FGSM Attack" or "Apply BIM Attack" to generate adversarial examples
   - View the results in the plot and performance metrics

## Features

- **Stock Selection**: Choose from available stock data in the data/raw directory
- **Attack Parameters**: Adjust the epsilon value to control attack strength
- **Visualization**: Compare original and perturbed time series
- **Performance Metrics**: View model accuracy and prediction change rates

## Requirements

- PyQt5
- matplotlib
- pandas
- numpy
- torch

## Data Requirements

The demo expects stock data in CSV format in the `data/raw` directory.