import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QComboBox, QPushButton, QLabel, 
                            QDoubleSpinBox, QGroupBox, QFormLayout, QMessageBox,
                            QSpinBox, QSplitter)
from PyQt5.QtCore import Qt
import torch

#grandparent = Path(__file__).resolve().parents[1]   # 0 = IOdemo, 1 = src, 2 = financial-adversarial-examples
#sys.path.append(str(grandparent))
src_path = Path(__file__).resolve().parents[1]
llama1_path = src_path.joinpath("lag_llama")
data_path = src_path.joinpath("data")
llama2_path = llama1_path.joinpath("lag_llama")
print("############################ PATH")
print(sys.path)
sys.path.append(str(src_path))
sys.path.append(str(llama1_path))
sys.path.append(str(llama2_path))
sys.path.append(str(data_path))
print(sys.path)
from attackedModelCpu import (load_data, load_data_llama, fgsm_attack, basic_iterative_method,generate_adversarial_data_cpu, generate_adversarial_llama_cpu,  evaluate_model)

class AdversarialDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Financial Adversarial Examples Demo")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.original_data = None
        self.perturbed_data = None
        self.current_stock = None
        self.epsilon = 4.54
        self.zoom_start = 0
        self.zoom_end = 100
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Stock selection
        stock_group = QGroupBox("Stock Selection")
        stock_layout = QFormLayout()
        self.stock_combo = QComboBox()
        self.load_stock_list()
        stock_layout.addRow("Select Stock:", self.stock_combo)
        stock_group.setLayout(stock_layout)
        left_layout.addWidget(stock_group)
        
        # Attack parameters
        attack_group = QGroupBox("Attack Parameters")
        attack_layout = QFormLayout()
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.1, 20.0)
        self.epsilon_spin.setSingleStep(0.1)
        self.epsilon_spin.setValue(self.epsilon)
        attack_layout.addRow("Epsilon:", self.epsilon_spin)
        attack_group.setLayout(attack_layout)
        left_layout.addWidget(attack_group)
        
        # Attack buttons
        self.fgsm_button = QPushButton("Apply FGSM Attack")
        self.bim_button = QPushButton("Apply BIM Attack")
        self.fgsm_button.clicked.connect(lambda: self.apply_attack('fgsm'))
        self.bim_button.clicked.connect(lambda: self.apply_attack('bim'))
        left_layout.addWidget(self.fgsm_button)
        left_layout.addWidget(self.bim_button)
        
        # Evaluation results
        eval_group = QGroupBox("Model Performance")
        eval_layout = QFormLayout()
        self.original_acc_label = QLabel("Original Accuracy: --")
        self.perturbed_acc_label = QLabel("Perturbed Accuracy: --")
        self.change_rate_label = QLabel("Prediction Change Rate: --")
        eval_layout.addRow(self.original_acc_label)
        eval_layout.addRow(self.perturbed_acc_label)
        eval_layout.addRow(self.change_rate_label)
        eval_group.setLayout(eval_layout)
        left_layout.addWidget(eval_group)
        
        # Zoom controls
        zoom_group = QGroupBox("Zoom Controls")
        zoom_layout = QFormLayout()
        self.zoom_start_spin = QSpinBox()
        self.zoom_end_spin = QSpinBox()
        self.zoom_start_spin.setRange(0, 1000)
        self.zoom_end_spin.setRange(0, 1000)
        self.zoom_start_spin.setValue(self.zoom_start)
        self.zoom_end_spin.setValue(self.zoom_end)
        self.zoom_start_spin.valueChanged.connect(self.update_zoom)
        self.zoom_end_spin.valueChanged.connect(self.update_zoom)
        zoom_layout.addRow("Start Index:", self.zoom_start_spin)
        zoom_layout.addRow("End Index:", self.zoom_end_spin)
        zoom_group.setLayout(zoom_layout)
        left_layout.addWidget(zoom_group)
        
        # Add stretch to push everything to the top
        left_layout.addStretch()
        
        # Create right panel for plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create right panel for plots
        right2_panel = QWidget()
        right2_layout = QVBoxLayout(right2_panel)
        
        # Create matplotlib figure with two subplots
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        # Create another matplotlib figure with two subplots
        self.figure2 = Figure(figsize=(8, 6))
        self.canvas2 = FigureCanvas(self.figure2)
        right2_layout.addWidget(self.canvas2)
        
        # Add panels to main layout
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)
        layout.addWidget(right2_panel, 2)
        
        # Connect signals
        self.stock_combo.currentTextChanged.connect(self.load_stock_data)
        self.epsilon_spin.valueChanged.connect(self.update_epsilon)
        
        # Load initial data
        if self.stock_combo.count() > 0:
            self.load_stock_data(self.stock_combo.currentText())
    
    def load_stock_list(self):
        """Load available stocks from data/raw directory"""
        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            QMessageBox.critical(self, "Error", "data/raw directory not found!")
            return
        
        for file in raw_dir.glob("*.csv"):
            self.stock_combo.addItem(str(file.stem).replace(".csv",""))
    
    def load_stock_data(self, stock_name):
        """Load and display the selected stock's data"""
        if not stock_name:
            return
        
        try:
            '''
            # First, process the new stock data
            from prepareData import splitTrainTest
            import os
            
            # Define paths
            raw_path = os.path.join("data", "raw")
            processed_dir = os.path.join("data", "processed")
            os.makedirs(processed_dir, exist_ok=True)
            processed_path = os.path.join(processed_dir, "cleaned_data.csv")
            
            # Process the new stock data
            train_df, eval_df = splitTrainTest(raw_path, f"{stock_name}.csv", processed_path)
            
            # Read the processed data for display
            file_path = Path("data/raw") / f"{stock_name}.csv"
            '''
            file_path = Path("data/processed") / f"cleaned_data_{stock_name}.csv"
            df = pd.read_csv(file_path)
            
            # Store the data
            self.current_stock = stock_name
            self.current_stock_file = file_path
            self.original_data = df['Close'].values
            
            # Update zoom range
            self.zoom_start_spin.setRange(0, len(self.original_data) - 1)
            self.zoom_end_spin.setRange(0, len(self.original_data) - 1)
            self.zoom_end_spin.setValue(min(100, len(self.original_data) - 1))
            
            # Plot the data
            self.plot_data()
            self.plot_data2()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load stock data: {str(e)}")
    
    def update_epsilon(self, value):
        """Update the epsilon value"""
        self.epsilon = value
    
    def apply_attack(self, attack_type):
        """Apply the selected attack to the current stock"""
        if self.original_data is None:
            QMessageBox.warning(self, "Warning", "Please select a stock first!")
            return
        
        try:
            # Generate adversarial data
            generate_adversarial_data_cpu(
                epsilon=self.epsilon,
                type=attack_type,
                dataID=self.current_stock
            )

            generate_adversarial_llama_cpu(
                epsilon=self.epsilon,
                type=attack_type,
                dataID=self.current_stock
            )
            
            # Load the perturbed data
            self.perturbed_data, _ = load_data(attacked=True, type=attack_type, dataID=self.current_stock)
            self.perturbed_data2, _ = load_data(attacked=True, type=attack_type, dataID=self.current_stock) # we are not calling llama
            
            # Plot both original and perturbed data
            self.plot_data(show_perturbed=True)
            self.plot_data2(show_perturbed=True)
            
            # Evaluate model performance
            self.evaluate_performance(attack_type)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply attack: {str(e)}")
    
    def update_zoom(self):
        """Update the zoom range and redraw the plot"""
        self.zoom_start = self.zoom_start_spin.value()
        self.zoom_end = self.zoom_end_spin.value()
        if self.zoom_start >= self.zoom_end:
            self.zoom_end = self.zoom_start + 1
            self.zoom_end_spin.setValue(self.zoom_end)
        self.plot_data(show_perturbed=self.perturbed_data is not None)
        self.plot_data2(show_perturbed=self.perturbed_data2 is not None)
    
    def plot_data(self, show_perturbed=False):
        """Plot the time series data"""
        self.figure.clear()
        
        # Create two subplots: one for the full view and one for the zoomed view
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        if self.original_data is None:
            return
        
        # Plot full view
        ax1.plot(self.original_data, label='Original', alpha=0.7)
        if show_perturbed and self.perturbed_data is not None:
            ax1.plot(self.perturbed_data, label='Perturbed', alpha=0.7)
        ax1.set_title(f"Full View - Stock Price: {self.current_stock}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot zoomed view
        start_idx = max(0, min(self.zoom_start, len(self.original_data)))
        end_idx = min(len(self.original_data), max(self.zoom_end, start_idx + 1))
        
        ax2.plot(range(start_idx, end_idx), 
                self.original_data[start_idx:end_idx], 
                label='Original', alpha=0.7)
        if show_perturbed and self.perturbed_data is not None:
            ax2.plot(range(start_idx, end_idx), 
                    self.perturbed_data[start_idx:end_idx], 
                    label='Perturbed', alpha=0.7)
        ax2.set_title(f"Zoomed View (Indices {start_idx}-{end_idx})")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_data2(self, show_perturbed=False):
        """Plot the time series data"""
        self.figure2.clear()
        
        # Create two subplots: one for the full view and one for the zoomed view
        ax1 = self.figure2.add_subplot(211)
        ax2 = self.figure2.add_subplot(212)
        
        if self.original_data is None:
            return
        
        # Plot full view
        ax1.plot(self.original_data, label='Original', alpha=0.7)
        if show_perturbed and self.perturbed_data2 is not None:
            ax1.plot(self.perturbed_data2, label='Perturbed', alpha=0.7)
        ax1.set_title(f"Full View - Stock Price: {self.current_stock}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot zoomed view
        start_idx = max(0, min(self.zoom_start, len(self.original_data)))
        end_idx = min(len(self.original_data), max(self.zoom_end, start_idx + 1))
        
        ax2.plot(range(start_idx, end_idx), 
                self.original_data[start_idx:end_idx], 
                label='Original', alpha=0.7)
        if show_perturbed and self.perturbed_data2 is not None:
            ax2.plot(range(start_idx, end_idx), 
                    self.perturbed_data2[start_idx:end_idx], 
                    label='Perturbed', alpha=0.7)
        ax2.set_title(f"Zoomed View (Indices {start_idx}-{end_idx})")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        self.figure2.tight_layout()

        self.canvas2.draw()
    
    def evaluate_performance(self, attack_type):
        """Evaluate model performance on original and perturbed data"""
        try:
            # Evaluate on original data
            original_labels, original_preds, _ = evaluate_model(attacked=False, dataID=self.current_stock)
            original_accuracy = sum(1 for x, y in zip(original_labels, original_preds) if x == y) / len(original_labels)
            
            # Evaluate on perturbed data
            perturbed_labels, perturbed_preds, _ = evaluate_model(attacked=True, type=attack_type, dataID=self.current_stock)
            perturbed_accuracy = sum(1 for x, y in zip(perturbed_labels, perturbed_preds) if x == y) / len(perturbed_labels)
            
            # Calculate prediction change rate
            prediction_changes = sum(1 for x, y in zip(original_preds, perturbed_preds) if x != y)
            change_rate = prediction_changes / len(original_preds)
            
            # Update labels
            self.original_acc_label.setText(f"Original Accuracy: {original_accuracy:.2%}")
            self.perturbed_acc_label.setText(f"Perturbed Accuracy: {perturbed_accuracy:.2%}")
            self.change_rate_label.setText(f"Prediction Change Rate: {change_rate:.2%}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to evaluate performance: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = AdversarialDemo()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 