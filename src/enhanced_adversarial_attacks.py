import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import the CNN model from the model.py file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lag_llama.model import CNN1DModel2, load_prescaled_regression_data

class AdversarialTimeSeriesAttacker:
    """
    Advanced adversarial input generator for forcing CNN regression models
    to predict steep, continuous downward curves.
    """
    
    def __init__(self, model, window_size=28, pred_horizon=14, device='cpu'):
        """
        Initialize the adversarial attacker.
        
        Args:
            model: Pretrained CNN1DModel2
            window_size: Input sequence length
            pred_horizon: Number of future predictions
            device: Computation device
        """
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        self.device = device
        
        # Disable gradient computation for model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def create_declining_target(self, decline_type='parabolic', start_value=0.5, 
                              end_value=None, slope=-0.5, acceleration=-0.05, 
                              decay_rate=0.15):
        """
        Create a target sequence with various decline patterns.
        
        Args:
            decline_type: Type of decline ('linear', 'parabolic', 'exponential')
            start_value: Starting prediction value
            end_value: Ending prediction value (auto-calculated if None)
            slope: Initial slope for linear/parabolic decline
            acceleration: Acceleration parameter for parabolic decline (negative for steeper curve)
            decay_rate: Decay rate for exponential decline
            
        Returns:
            torch.Tensor: Target declining sequence
        """
        timesteps = torch.arange(self.pred_horizon, dtype=torch.float32, device=self.device)
        
        if decline_type == 'linear':
            # Linear decline: y = start_value + slope * t
            target = start_value + slope * timesteps
            
        elif decline_type == 'parabolic':
            # Parabolic decline: y = start_value + slope * t + acceleration * t^2
            target = start_value + slope * timesteps + acceleration * (timesteps ** 2)
            
        elif decline_type == 'exponential':
            # Exponential decline: y = start_value * exp(-decay_rate * t)
            # Adjust to reach a reasonable end value
            if end_value is None:
                end_value = start_value * 0.1  # Default to 10% of start value
            
            # Calculate decay rate to reach end_value at final timestep
            if self.pred_horizon > 1:
                decay_rate = -torch.log(torch.tensor(end_value / start_value)) / (self.pred_horizon - 1)
            
            target = start_value * torch.exp(-decay_rate * timesteps)
            
        else:
            raise ValueError(f"Unknown decline_type: {decline_type}. Use 'linear', 'parabolic', or 'exponential'")
        
        # Ensure it stays within reasonable bounds
        target = torch.clamp(target, min=-2.0, max=2.0)
        
        return target.unsqueeze(0)  # Add batch dimension
    
    def adversarial_loss(self, predictions, target_decline, steepness_factor=3.0):
        """
        Comprehensive loss function to force steep downward predictions.
        
        Args:
            predictions: Model predictions (batch_size, pred_horizon)
            target_decline: Target declining sequence
            steepness_factor: Amplification factor for negative slopes
            
        Returns:
            torch.Tensor: Total adversarial loss
        """
        # Primary MSE loss to match target declining sequence
        mse_loss = F.mse_loss(predictions, target_decline)
        
        # Calculate slopes between consecutive predictions
        if predictions.size(1) > 1:
            slopes = predictions[:, 1:] - predictions[:, :-1]
            
            # Heavily penalize any positive slopes (upward movement)
            positive_slope_penalty = torch.sum(torch.clamp(slopes, min=0.0) ** 2) * 10.0
            
            # Reward negative slopes (amplify steepness)
            negative_slope_reward = -torch.sum(torch.clamp(slopes, max=0.0) ** 2) * steepness_factor
            
            slope_loss = positive_slope_penalty + negative_slope_reward
        else:
            slope_loss = 0.0
        
        # Additional penalty for predictions that are too high
        high_value_penalty = torch.sum(torch.clamp(predictions - 0.0, min=0.0) ** 2) * 2.0
        
        # Total loss combines all components
        total_loss = mse_loss + slope_loss + high_value_penalty
        
        return total_loss, mse_loss, slope_loss, high_value_penalty
    
    def generate_adversarial_input(self, original_input, decline_type='parabolic',
                                 target_slope=-0.75, acceleration=-0.05,
                                 iterations=1500, learning_rate=0.01, 
                                 momentum=0.9, input_bounds=(-0.1, 1.1),
                                 max_perturbation_ratio=0.10):
        """
        Generate adversarial input using gradient-based optimization with momentum.
        
        Args:
            original_input: Original time series input (batch_size, 1, window_size)
            decline_type: Type of decline ('linear', 'parabolic', 'exponential')
            target_slope: Target initial slope for decline
            acceleration: Acceleration parameter for parabolic decline
            iterations: Number of optimization iterations
            learning_rate: Optimization learning rate
            momentum: Momentum factor for updates
            input_bounds: Clamping bounds for inputs
            max_perturbation_ratio: Maximum perturbation as ratio of input (default: 0.10 = 10%)
            
        Returns:
            torch.Tensor: Adversarial input
            dict: Attack metrics and history
        """
        # Clone and prepare adversarial input
        adv_input = original_input.clone().to(self.device)
        adv_input.requires_grad_(True)
        
        # Create target declining sequence
        target_decline = self.create_declining_target(
            decline_type=decline_type,
            start_value=original_input.mean().item(),
            slope=target_slope,
            acceleration=acceleration
        )
        
        # Initialize momentum
        momentum_buffer = torch.zeros_like(adv_input)
        
        # Calculate maximum allowed perturbation for each input element
        # Use 10% of absolute input values, with minimum threshold for very small values
        min_perturbation_threshold = 0.01  # Minimum allowed perturbation
        max_perturbation = torch.maximum(
            torch.abs(original_input) * max_perturbation_ratio,
            torch.full_like(original_input, min_perturbation_threshold)
        )
        
        # Track optimization history
        history = {
            'losses': [],
            'slopes': [],
            'predictions': [],
            'input_changes': [],
            'max_perturbations': []
        }
        
        print(f"Starting adversarial optimization for {iterations} iterations...")
        print(f"Decline type: {decline_type}")
        print(f"Target slope: {target_slope}")
        if decline_type == 'parabolic':
            print(f"Acceleration: {acceleration}")
        print(f"Learning rate: {learning_rate}, Momentum: {momentum}")
        print(f"Max perturbation ratio: {max_perturbation_ratio * 100:.1f}%")
        
        for i in range(iterations):
            # Forward pass
            predictions = self.model(adv_input)
            
            # Calculate adversarial loss
            total_loss, mse_loss, slope_loss, penalty_loss = self.adversarial_loss(
                predictions, target_decline, steepness_factor=3.0)
            
            # Backward pass
            total_loss.backward()
            
            # Apply gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_([adv_input], max_norm=1.0)
            
            # Momentum-based update
            with torch.no_grad():
                # Update momentum buffer
                momentum_buffer = momentum * momentum_buffer + adv_input.grad
                
                # Apply update
                adv_input -= learning_rate * momentum_buffer
                
                # Clamp to bounds
                adv_input.clamp_(input_bounds[0], input_bounds[1])
                
                # Apply 10% perturbation constraint
                # Calculate current perturbation
                current_perturbation = adv_input - original_input
                
                # Clamp perturbation to maximum allowed values
                clamped_perturbation = torch.clamp(
                    current_perturbation,
                    min=-max_perturbation,
                    max=max_perturbation
                )
                
                # Update adversarial input with clamped perturbation (in-place to preserve gradients)
                adv_input.copy_(original_input + clamped_perturbation)
                
                # Ensure final result still respects input bounds
                adv_input.clamp_(input_bounds[0], input_bounds[1])
                
                # Track metrics
                if i % 100 == 0 or i == iterations - 1:
                    current_slopes = self._calculate_slopes(predictions)
                    avg_slope = current_slopes.mean().item()
                    min_slope = current_slopes.min().item()
                    max_slope = current_slopes.max().item()
                    
                    # Calculate actual perturbation statistics
                    actual_perturbation = torch.abs(adv_input - original_input)
                    max_actual_perturbation = actual_perturbation.max().item()
                    avg_perturbation_ratio = (actual_perturbation / (torch.abs(original_input) + 1e-8)).mean().item()
                    
                    history['losses'].append(total_loss.item())
                    history['slopes'].append(avg_slope)
                    history['predictions'].append(predictions.clone())
                    history['input_changes'].append(torch.norm(adv_input - original_input).item())
                    history['max_perturbations'].append(max_actual_perturbation)
                    
                    print(f"Iter {i:4d}: Loss={total_loss.item():.4f}, "
                          f"Avg Slope={avg_slope:.4f}, Min Slope={min_slope:.4f}, "
                          f"Max Slope={max_slope:.4f}, Max Pert={max_actual_perturbation:.4f}, "
                          f"Avg Pert Ratio={avg_perturbation_ratio*100:.2f}%")
            
            # Clear gradients
            if adv_input.grad is not None:
                adv_input.grad.zero_()
        
        return adv_input.detach(), history
    
    def _calculate_slopes(self, predictions):
        """Calculate slopes between consecutive predictions."""
        if predictions.size(1) > 1:
            return predictions[:, 1:] - predictions[:, :-1]
        else:
            return torch.zeros(predictions.size(0), 1, device=predictions.device)

def load_pretrained_model(model_path, pred_horizon=14, device='cpu'):
    """
    Load a pretrained CNN1DModel2 from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        pred_horizon: Prediction horizon
        device: Computation device
        
    Returns:
        CNN1DModel2: Loaded model
    """
    model = CNN1DModel2(pred_horizon=pred_horizon)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def create_sample_input(window_size=28, device='cpu', use_real_data=True, stock_ticker='GOOGL'):
    """
    Create sample input data for adversarial attacks using evaluation_data.csv.
    
    Args:
        window_size: Input sequence length
        device: Computation device
        use_real_data: Whether to use real stock data
        stock_ticker: Stock ticker for real data (unused, kept for compatibility)
        
    Returns:
        torch.Tensor: Sample input data
        list: Additional metadata (dates)
    """
    if use_real_data:
        try:
            # Load from the specified evaluation data file
            examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            evaluation_data_path = os.path.join(examples_dir, "data", "processed", "evaluation_data.csv")
            
            if not os.path.exists(evaluation_data_path):
                raise FileNotFoundError(f"Evaluation data not found at {evaluation_data_path}")
                
            # Load the evaluation data
            df = pd.read_csv(evaluation_data_path)
            print(f"Loaded evaluation data with {len(df)} rows")
            print(f"Columns available: {list(df.columns)}")
            
            # Check for required columns
            if 'Close' not in df.columns:
                raise ValueError(f"'Close' column not found in evaluation data. Available columns: {list(df.columns)}")
            
            # Extract close prices
            prices = df['Close'].values
            
            # Get dates if available
            if 'Date' in df.columns:
                dates = pd.to_datetime(df['Date'])
            else:
                dates = pd.date_range(start='2018-01-01', periods=len(prices), freq='D')
            
            # Ensure we have enough data for the window
            if len(prices) < window_size:
                raise ValueError(f"Not enough data points ({len(prices)}) for window size ({window_size})")
            
            # Take a random window from the available data
            max_start = len(prices) - window_size
            start_idx = np.random.randint(0, max_start + 1)
            sample_data = prices[start_idx:start_idx + window_size]
            sample_dates = dates[start_idx:start_idx + window_size]
            
            # Normalize the data (min-max scaling to [0,1] range like the trained model expects)
            data_min = sample_data.min()
            data_max = sample_data.max()
            if data_max > data_min:
                sample_data_normalized = (sample_data - data_min) / (data_max - data_min)
            else:
                sample_data_normalized = sample_data - data_min  # All values are the same
            
            print(f"Selected data window from index {start_idx} to {start_idx + window_size - 1}")
            print(f"Original price range: ${data_min:.2f} - ${data_max:.2f}")
            print(f"Normalized range: {sample_data_normalized.min():.4f} - {sample_data_normalized.max():.4f}")
            
            # Convert to tensor format (batch_size=1, channels=1, window_size)
            input_tensor = torch.tensor(sample_data_normalized, dtype=torch.float32, device=device)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            
            return input_tensor, sample_dates.tolist()
            
        except Exception as e:
            print(f"Could not load evaluation data: {e}")
            print("Falling back to synthetic data...")
    
    # Generate synthetic time series data as fallback
    print("Generating synthetic time series data...")
    np.random.seed(42)
    
    # Create a realistic-looking time series with trend and noise
    t = np.linspace(0, 1, window_size)
    trend = 0.1 * t  # Slight upward trend
    noise = 0.05 * np.random.randn(window_size)
    seasonal = 0.02 * np.sin(2 * np.pi * 4 * t)  # Some seasonality
    
    synthetic_data = 0.5 + trend + seasonal + noise  # Normalize around 0.5
    
    # Convert to tensor format
    input_tensor = torch.tensor(synthetic_data, dtype=torch.float32, device=device)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    
    # Generate synthetic dates
    synthetic_dates = pd.date_range(start='2018-01-01', periods=window_size, freq='D')
    
    return input_tensor, synthetic_dates.tolist()

def visualize_attack_results(original_input, adversarial_input, original_pred, 
                           adversarial_pred, history, save_path=None):
    """
    Create visualization showing the complete timeline: input → predictions.
    
    Args:
        original_input: Original input sequence
        adversarial_input: Adversarial input sequence  
        original_pred: Original model predictions
        adversarial_pred: Adversarial model predictions
        history: Attack optimization history
        save_path: Path to save plots
    """
    # Convert tensors to numpy for plotting
    orig_input = original_input.squeeze().cpu().numpy()
    adv_input = adversarial_input.squeeze().cpu().numpy()
    orig_pred = original_pred.squeeze().cpu().numpy()
    adv_pred = adversarial_pred.squeeze().cpu().numpy()
    
    # Create a single large figure for the timeline
    plt.figure(figsize=(16, 8))
    
    # Combined timeline: Input → Predictions
    total_steps = len(orig_input) + len(orig_pred)
    timeline_orig = np.concatenate([orig_input, orig_pred])
    timeline_adv = np.concatenate([adv_input, adv_pred])
    
    plt.plot(range(total_steps), timeline_orig, 'b-', 
             label='Original (Input+Predictions)', linewidth=3, alpha=0.7)
    plt.plot(range(total_steps), timeline_adv, 'r-', 
             label='Adversarial (Input+Predictions)', linewidth=3)
    plt.axvline(x=len(orig_input), color='black', linestyle=':', 
               linewidth=2, label='Input|Prediction Boundary')
    
    plt.title('Complete Timeline: Input → Predictions', fontsize=18, fontweight='bold')
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some styling improvements
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def create_prediction_comparison_plot(original_pred, adversarial_pred, save_path=None):
    """
    Create a standalone plot focusing on original vs adversarial predictions comparison.
    
    Args:
        original_pred: Original model predictions
        adversarial_pred: Adversarial model predictions
        save_path: Path to save the plot
    """
    # Convert tensors to numpy for plotting
    orig_pred = original_pred.squeeze().cpu().numpy()
    adv_pred = adversarial_pred.squeeze().cpu().numpy()
    
    # Create a new figure
    plt.figure(figsize=(12, 6))
    
    pred_steps = range(len(orig_pred))
    
    # Plot both prediction series
    plt.plot(pred_steps, orig_pred, 'b-o', label='Original Predictions', 
             linewidth=3, markersize=6)
    plt.plot(pred_steps, adv_pred, 'r-s', label='Adversarial Predictions', 
             linewidth=3, markersize=6)
    
    # Add a trend line for adversarial predictions
    if len(adv_pred) > 1:
        x_trend = np.array(pred_steps)
        z = np.polyfit(x_trend, adv_pred, 1)
        p = np.poly1d(z)
        plt.plot(x_trend, p(x_trend), 'r--', alpha=0.7, linewidth=2, 
                label=f'Adversarial Trend (slope: {z[0]:.4f})')
    
    # Styling
    plt.title('Original vs Adversarial Model Predictions', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Prediction Step', fontsize=12)
    plt.ylabel('Predicted Value (Normalized)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Standalone prediction comparison saved to: {save_path}")
    
    plt.show()

def create_contextual_timeline_plot(model, original_input, adversarial_input, 
                                  original_pred, adversarial_pred, save_path=None):
    """
    Create a contextual timeline showing 28 days of original predictions 
    with the last 14 days replaced by adversarial predictions.
    
    Args:
        model: The CNN model
        original_input: Original input sequence
        adversarial_input: Adversarial input sequence
        original_pred: Original predictions (14 days)
        adversarial_pred: Adversarial predictions (14 days)
        save_path: Path to save the plot
    """
    device = original_input.device
    window_size = original_input.size(-1)  # Should be 28
    pred_horizon = original_pred.size(-1)  # Should be 14
    
    # Generate extended original predictions (28 days total)
    # We'll slide the window to get 28 predictions
    orig_input_np = original_input.squeeze().cpu().numpy()
    extended_original_preds = []
    
    with torch.no_grad():
        for i in range(pred_horizon + 1):  # Generate 15 sets of predictions
            if i == 0:
                # Use original input for first prediction
                current_input = original_input
            else:
                # Slide window forward using previous predictions
                # Take last (window_size-1) from input + i predictions
                if i <= len(orig_input_np):
                    new_sequence = np.concatenate([
                        orig_input_np[i:],
                        extended_original_preds[0][:i]
                    ])
                else:
                    # Use previous predictions to fill the window
                    start_idx = max(0, i - window_size)
                    pred_sequence = np.concatenate(extended_original_preds[start_idx:i])
                    if len(pred_sequence) >= window_size:
                        new_sequence = pred_sequence[-window_size:]
                    else:
                        remaining = window_size - len(pred_sequence)
                        new_sequence = np.concatenate([
                            orig_input_np[-remaining:],
                            pred_sequence
                        ])
                
                current_input = torch.tensor(new_sequence, dtype=torch.float32, device=device)
                current_input = current_input.unsqueeze(0).unsqueeze(0)
            
            # Get prediction for current input
            pred = model(current_input)
            extended_original_preds.append(pred.squeeze().cpu().numpy())
    
    # Create the contextual timeline
    # First 14 days: original predictions
    # Last 14 days: adversarial predictions
    timeline_original = np.concatenate(extended_original_preds[:2])  # 28 days
    timeline_hybrid = np.concatenate([
        extended_original_preds[0],  # First 14 days original
        adversarial_pred.squeeze().cpu().numpy()  # Last 14 days adversarial
    ])
    
    # Create the plot
    plt.figure(figsize=(14, 6))
    
    timeline_days = range(len(timeline_original))
    transition_point = pred_horizon
    
    # Plot full original timeline
    plt.plot(timeline_days, timeline_original, 'b-o', 
             label='Original Predictions (Full Timeline)', 
             linewidth=2, markersize=4, alpha=0.7)
    
    # Plot hybrid timeline (original + adversarial)
    plt.plot(timeline_days[:transition_point], timeline_hybrid[:transition_point], 
             'b-o', linewidth=3, markersize=5, alpha=1.0)
    plt.plot(timeline_days[transition_point:], timeline_hybrid[transition_point:], 
             'r-s', label='Adversarial Predictions (Attack Effect)', 
             linewidth=3, markersize=5)
    
    # Add vertical line to mark the attack point
    plt.axvline(x=transition_point, color='orange', linestyle=':', 
               linewidth=2, label='Attack Transition Point')
    
    plt.title('Contextual Timeline: Original vs Adversarial Attack Effect', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Day', fontsize=12)
    plt.ylabel('Predicted Value (Normalized)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Contextual timeline plot saved to: {save_path}")
    
    plt.show()

def calculate_attack_metrics(original_pred, adversarial_pred):
    """
    Calculate comprehensive metrics for attack success.
    
    Args:
        original_pred: Original model predictions
        adversarial_pred: Adversarial model predictions
        
    Returns:
        dict: Comprehensive metrics
    """
    orig_pred = original_pred.squeeze().cpu().numpy()
    adv_pred = adversarial_pred.squeeze().cpu().numpy()
    
    # Calculate slopes
    if len(adv_pred) > 1:
        adv_slopes = np.diff(adv_pred)
        orig_slopes = np.diff(orig_pred)
    else:
        adv_slopes = np.array([0])
        orig_slopes = np.array([0])
    
    metrics = {
        # Slope metrics for adversarial predictions
        'adversarial_avg_slope': np.mean(adv_slopes),
        'adversarial_min_slope': np.min(adv_slopes),
        'adversarial_max_slope': np.max(adv_slopes),
        'adversarial_slope_std': np.std(adv_slopes),
        
        # Slope metrics for original predictions
        'original_avg_slope': np.mean(orig_slopes),
        'original_min_slope': np.min(orig_slopes),
        'original_max_slope': np.max(orig_slopes),
        
        # Attack success metrics
        'total_decline': adv_pred[-1] - adv_pred[0] if len(adv_pred) > 1 else 0,
        'percent_negative_slopes': np.mean(adv_slopes < 0) * 100,
        'steepest_drop': np.min(adv_slopes),
        'prediction_range': np.max(adv_pred) - np.min(adv_pred),
        
        # Comparison metrics
        'slope_change': np.mean(adv_slopes) - np.mean(orig_slopes),
        'prediction_shift': np.mean(adv_pred) - np.mean(orig_pred)
    }
    
    return metrics

def print_attack_summary(metrics, target_slope=-0.5):
    """
    Print a comprehensive summary of attack results.
    
    Args:
        metrics: Attack metrics dictionary
        target_slope: Target slope for evaluation
    """
    print("\n" + "="*80)
    print(" ADVERSARIAL ATTACK RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nSLOPE ANALYSIS:")
    print(f"   • Average Slope: {metrics['adversarial_avg_slope']:.4f}")
    print(f"   • Minimum Slope: {metrics['adversarial_min_slope']:.4f}")  
    print(f"   • Maximum Slope: {metrics['adversarial_max_slope']:.4f}")
    print(f"   • Slope Std Dev: {metrics['adversarial_slope_std']:.4f}")
    
    print(f"\nTARGET ACHIEVEMENT:")
    target_met = metrics['adversarial_avg_slope'] <= target_slope
    print(f"   • Target Slope: {target_slope}")
    print(f"   • Target Met: {'YES' if target_met else 'NO'}")
    print(f"   • Steepest Drop: {metrics['steepest_drop']:.4f}")
    
    print(f"\nDECLINE CHARACTERISTICS:")
    print(f"   • Total Decline: {metrics['total_decline']:.4f}")
    print(f"   • Percent Negative Slopes: {metrics['percent_negative_slopes']:.1f}%")
    print(f"   • Prediction Range: {metrics['prediction_range']:.4f}")
    
    print(f"\nCOMPARISON WITH ORIGINAL:")
    print(f"   • Original Avg Slope: {metrics['original_avg_slope']:.4f}")
    print(f"   • Slope Change: {metrics['slope_change']:.4f}")
    print(f"   • Prediction Shift: {metrics['prediction_shift']:.4f}")
    
    print(f"\nSUCCESS CRITERIA:")
    print(f"   • Continuous Decline: {'PASS' if metrics['percent_negative_slopes'] >= 80 else 'FAIL'} "
          f"({metrics['percent_negative_slopes']:.1f}% negative slopes)")
    print(f"   • Steep Enough: {'PASS' if target_met else 'FAIL'} "
          f"(avg slope {metrics['adversarial_avg_slope']:.4f} vs target {target_slope})")
    print(f"   • No Major Reversals: {'PASS' if metrics['adversarial_max_slope'] < 0.1 else 'FAIL'} "
          f"(max slope: {metrics['adversarial_max_slope']:.4f})")
    
    # Overall success assessment
    overall_success = (target_met and 
                      metrics['percent_negative_slopes'] >= 80 and 
                      metrics['adversarial_max_slope'] < 0.1)
    
    print(f"\nOVERALL SUCCESS: {'EXCELLENT' if overall_success else 'NEEDS IMPROVEMENT'}")
    
    print("="*80)

def main():
    """
    Main function to demonstrate complete adversarial attack pipeline.
    """
    # Configuration
    config = {
        'window_size': 28,
        'pred_horizon': 14,
        'model_path': 'experiments/models/GOOGL_regression_model_14day.pth',
        'target_slope': -0.75,  # Very steep decline
        'iterations': 1500,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'max_perturbation_ratio': 0.10,  # 10% max perturbation
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("ENHANCED ADVERSARIAL TIME SERIES ATTACK")
    print("="*60)
    print(f"Device: {config['device']}")
    print(f"Model: {config['model_path']}")
    print(f"Target Slope: {config['target_slope']}")
    print(f"Max Perturbation: {config['max_perturbation_ratio']*100:.0f}%")
    print(f"Iterations: {config['iterations']}")
    print(f"Data Source: ../data/processed/evaluation_data.csv")
    
    try:
        # Load pretrained model
        print(f"\nLoading pretrained model...")
        model = load_pretrained_model(config['model_path'], 
                                    config['pred_horizon'], 
                                    config['device'])
        print("Model loaded successfully")
        
        # Create adversarial attacker
        attacker = AdversarialTimeSeriesAttacker(
            model=model,
            window_size=config['window_size'],
            pred_horizon=config['pred_horizon'],
            device=config['device']
        )
        
        # Generate sample input from evaluation data
        print(f"\nLoading sample input data from evaluation_data.csv...")
        original_input, metadata = create_sample_input(
            window_size=config['window_size'],
            device=config['device'],
            use_real_data=True
        )
        print("Sample data loaded from evaluation file")
        
        # Get original predictions
        print(f"\nGetting original model predictions...")
        with torch.no_grad():
            original_predictions = model(original_input)
        print("Original predictions obtained")
        
        # Generate adversarial input
        print(f"\nLaunching adversarial attack...")
        adversarial_input, attack_history = attacker.generate_adversarial_input(
            original_input=original_input,
            decline_type='parabolic',  # Use parabolic decline
            target_slope=config['target_slope'],
            acceleration=-0.08,  # Steep parabolic acceleration
            iterations=config['iterations'],
            learning_rate=config['learning_rate'],
            momentum=config['momentum'],
            max_perturbation_ratio=config['max_perturbation_ratio']
        )
        print("Adversarial attack completed")
        
        # Get adversarial predictions
        print(f"\nGetting adversarial model predictions...")
        with torch.no_grad():
            adversarial_predictions = model(adversarial_input)
        print("Adversarial predictions obtained")
        
        # Calculate metrics
        print(f"\nCalculating attack metrics...")
        metrics = calculate_attack_metrics(original_predictions, adversarial_predictions)
        print("Metrics calculated")
        
        # Print comprehensive summary
        print_attack_summary(metrics, config['target_slope'])
        
        # Create visualization with perturbation percentage in filename
        print(f"\nCreating visualizations...")
        pert_pct = int(config['max_perturbation_ratio'] * 100)
        save_path = os.path.join('experiments', 'plots', 
                                f'complete_adversarial_attack_evaluation_data_pert_{pert_pct}pct.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        visualize_attack_results(
            original_input=original_input,
            adversarial_input=adversarial_input,
            original_pred=original_predictions,
            adversarial_pred=adversarial_predictions,
            history=attack_history,
            save_path=save_path
        )
        print("Visualization completed")

        # Create standalone prediction comparison plot
        print(f"\nCreating standalone prediction comparison...")
        standalone_save_path = os.path.join('experiments', 'plots', 
                                          f'predictions_comparison_evaluation_data_pert_{pert_pct}pct.png')
        create_prediction_comparison_plot(
            original_pred=original_predictions,
            adversarial_pred=adversarial_predictions,
            save_path=standalone_save_path
        )
        print("Standalone prediction comparison completed")
        
        # Create contextual timeline plot
        print(f"\nCreating contextual timeline plot...")
        contextual_save_path = os.path.join('experiments', 'plots', 
                                          f'contextual_timeline_evaluation_data_pert_{pert_pct}pct.png')
        create_contextual_timeline_plot(
            model=model,
            original_input=original_input,
            adversarial_input=adversarial_input,
            original_pred=original_predictions,
            adversarial_pred=adversarial_predictions,
            save_path=contextual_save_path
        )
        print("Contextual timeline plot completed")
        
        return {
            'original_input': original_input,
            'adversarial_input': adversarial_input,
            'original_predictions': original_predictions,
            'adversarial_predictions': adversarial_predictions,
            'metrics': metrics,
            'history': attack_history
        }
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the complete adversarial attack demonstration
    results = main()
    
    if results is not None:
        print(f"\nADVERSARIAL ATTACK COMPLETED SUCCESSFULLY!")
        print(f"Check the generated plots to see the steep downward curve.")
    else:
        print(f"\nATTACK FAILED - Please check the error messages above.")
