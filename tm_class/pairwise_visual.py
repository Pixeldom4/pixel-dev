import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import argparse
import os

def load_data(file_path):
    """Load TSV data from file."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def create_prediction_plots(df, output_dir=None, show_plots=True):
    """Create various plots to visualize prediction performance."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Prediction Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Scatter plot with perfect prediction line
    ax1 = axes[0, 0]
    ax1.scatter(df['y_true'], df['y_pred'], alpha=0.6, s=30)
    
    # Add perfect prediction line (y=x)
    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['y_true'], df['y_pred'])
    line = slope * df['y_true'] + intercept
    ax1.plot(df['y_true'], line, 'g-', linewidth=2, label=f'Regression Line (R²={r_value**2:.3f})')
    
    ax1.set_xlabel('y_true')
    ax1.set_ylabel('y_pred')
    ax1.set_title('y_true vs y_pred')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals plot
    ax2 = axes[0, 1]
    residuals = df['y_pred'] - df['y_true']
    ax2.scatter(df['y_true'], residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('y_true')
    ax2.set_ylabel('Residuals (y_pred - y_true)')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(df['y_true'], bins=30, alpha=0.7, label='y_true', density=True)
    ax3.hist(df['y_pred'], bins=30, alpha=0.7, label='y_pred', density=True)
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Q-Q plot for residuals
    ax4 = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot of Residuals')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(output_dir, 'prediction_analysis.png')}")
    
    # Show plot if requested
    if show_plots:
        plt.show()
    
    return fig

def create_correlation_plot(df, output_dir=None, show_plots=True):
    """Create a detailed correlation plot with statistics."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(df['y_true'], df['y_pred'], alpha=0.6, s=40, c=df.index, cmap='viridis')
    
    # Add perfect prediction line
    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    
    # Calculate and add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['y_true'], df['y_pred'])
    line_x = np.array([min_val, max_val])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'g-', linewidth=2, label=f'Regression Line')
    
    # Calculate metrics
    correlation = df['y_true'].corr(df['y_pred'])
    mae = np.mean(np.abs(df['y_true'] - df['y_pred']))
    rmse = np.sqrt(np.mean((df['y_true'] - df['y_pred'])**2))
    
    # Add statistics text box
    stats_text = f'''Statistics:
Correlation (r): {correlation:.4f}
R²: {r_value**2:.4f}
MAE: {mae:.4f}
RMSE: {rmse:.4f}
Slope: {slope:.4f}
Intercept: {intercept:.4f}
N: {len(df)}'''
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('y_true', fontsize=12)
    ax.set_ylabel('y_pred', fontsize=12)
    ax.set_title('Prediction vs True Values with Statistics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Data Point Index', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'correlation_plot.png'), dpi=300, bbox_inches='tight')
        print(f"Correlation plot saved to {os.path.join(output_dir, 'correlation_plot.png')}")
    
    # Show plot if requested
    if show_plots:
        plt.show()
    
    return fig

def create_pairwise_difference_plot(df, output_dir=None, show_plots=True, max_lines=100):
    """Create plots showing pairwise connections between y_true and y_pred."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pairwise Difference Analysis', fontsize=16, fontweight='bold')
    
    # Subsample data if too many points for line plot
    if len(df) > max_lines:
        df_sample = df.sample(n=max_lines, random_state=42).sort_index()
        print(f"Note: Showing lines for {max_lines} randomly sampled points (out of {len(df)} total)")
    else:
        df_sample = df.copy()
    
    # Plot 1: Connected scatter plot with lines showing differences
    ax1 = axes[0, 0]
    
    # Plot points
    ax1.scatter(df['y_true'], [1]*len(df), alpha=0.6, s=30, color='blue', label='y_true')
    ax1.scatter(df['y_pred'], [2]*len(df), alpha=0.6, s=30, color='red', label='y_pred')
    
    # Draw lines connecting paired values (subsample for clarity)
    for idx in df_sample.index:
        y_true_val = df.loc[idx, 'y_true']
        y_pred_val = df.loc[idx, 'y_pred']
        
        # Color line by difference magnitude
        diff = abs(y_pred_val - y_true_val)
        color = plt.cm.viridis(min(diff / df['y_pred'].std(), 1.0))
        
        ax1.plot([y_true_val, y_pred_val], [1, 2], color=color, alpha=0.6, linewidth=0.8)
    
    ax1.set_ylim(0.5, 2.5)
    ax1.set_yticks([1, 2])
    ax1.set_yticklabels(['y_true', 'y_pred'])
    ax1.set_xlabel('Value')
    ax1.set_title('Paired Value Connections')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot with connecting lines
    ax2 = axes[0, 1]
    ax2.scatter(df['y_true'], df['y_pred'], alpha=0.6, s=40, c='blue', label='Data Points')
    
    # Draw lines from perfect prediction line to actual points (subsample)
    for idx in df_sample.index:
        y_true_val = df.loc[idx, 'y_true']
        y_pred_val = df.loc[idx, 'y_pred']
        
        # Color by error magnitude
        error = abs(y_pred_val - y_true_val)
        color = plt.cm.Reds(min(error / (2 * df['y_pred'].std()), 1.0))
        
        ax2.plot([y_true_val, y_true_val], [y_true_val, y_pred_val], 
                color=color, alpha=0.7, linewidth=1)
    
    # Perfect prediction line
    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('y_true')
    ax2.set_ylabel('y_pred')
    ax2.set_title('Vertical Distance to Perfect Line')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Difference magnitude plot
    ax3 = axes[1, 0]
    differences = df['y_pred'] - df['y_true']
    abs_differences = np.abs(differences)
    
    # Sort by index for connected line plot
    sorted_indices = sorted(df.index)
    ax3.plot(sorted_indices, differences[sorted_indices], 'o-', alpha=0.7, markersize=4, linewidth=1)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Data Point Index')
    ax3.set_ylabel('Difference (y_pred - y_true)')
    ax3.set_title('Prediction Errors by Index')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Difference distribution and statistics
    ax4 = axes[1, 1]
    ax4.hist(differences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax4.axvline(x=differences.mean(), color='green', linestyle='-', linewidth=2, 
               label=f'Mean Error: {differences.mean():.3f}')
    
    ax4.set_xlabel('Difference (y_pred - y_true)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Prediction Errors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'''Error Statistics:
Mean Error: {differences.mean():.3f}
Mean Abs Error: {abs_differences.mean():.3f}
Std Error: {differences.std():.3f}
Max Error: {abs_differences.max():.3f}
Min Error: {abs_differences.min():.3f}'''
    
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'pairwise_differences.png'), dpi=300, bbox_inches='tight')
        print(f"Pairwise difference plot saved to {os.path.join(output_dir, 'pairwise_differences.png')}")
    
    # Show plot if requested
    if show_plots:
        plt.show()
    
    return fig

def create_sign_analysis_plot(df, output_dir=None, show_plots=True):
    """Create plots focusing on sign agreement analysis."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Sign Agreement Analysis', fontsize=16, fontweight='bold')
    
    # Classify predictions by sign agreement
    true_positive = (df['y_true'] > 0) & (df['y_pred'] > 0)
    true_negative = (df['y_true'] < 0) & (df['y_pred'] < 0)
    false_positive = (df['y_true'] < 0) & (df['y_pred'] > 0)
    false_negative = (df['y_true'] > 0) & (df['y_pred'] < 0)
    
    # Plot 1: Colored scatter plot by classification
    ax1 = axes[0]
    ax1.scatter(df[true_positive]['y_true'], df[true_positive]['y_pred'], 
               alpha=0.7, s=40, color='green', label=f'True Positive ({true_positive.sum()})')
    ax1.scatter(df[true_negative]['y_true'], df[true_negative]['y_pred'], 
               alpha=0.7, s=40, color='blue', label=f'True Negative ({true_negative.sum()})')
    ax1.scatter(df[false_positive]['y_true'], df[false_positive]['y_pred'], 
               alpha=0.7, s=40, color='red', label=f'False Positive ({false_positive.sum()})')
    ax1.scatter(df[false_negative]['y_true'], df[false_negative]['y_pred'], 
               alpha=0.7, s=40, color='orange', label=f'False Negative ({false_negative.sum()})')
    
    # Add quadrant lines
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    
    ax1.set_xlabel('y_true')
    ax1.set_ylabel('y_pred')
    ax1.set_title('Classification by Sign Agreement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confusion matrix as heatmap
    ax2 = axes[1]
    confusion_data = np.array([[true_negative.sum(), false_positive.sum()],
                              [false_negative.sum(), true_positive.sum()]])
    
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'], ax=ax2)
    ax2.set_title('Confusion Matrix (Sign-based)')
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'sign_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Sign analysis plot saved to {os.path.join(output_dir, 'sign_analysis.png')}")
    
    # Show plot if requested
    if show_plots:
        plt.show()
    
    return fig

def print_summary_stats(df):
    """Print summary statistics for the data."""
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    correlation = df['y_true'].corr(df['y_pred'])
    mae = np.mean(np.abs(df['y_true'] - df['y_pred']))
    rmse = np.sqrt(np.mean((df['y_true'] - df['y_pred'])**2))
    
    # Sign agreement
    same_sign = ((df['y_true'] > 0) & (df['y_pred'] > 0)) | ((df['y_true'] < 0) & (df['y_pred'] < 0))
    directional_accuracy = same_sign.sum() / len(df)
    
    print(f"Total samples: {len(df)}")
    print(f"Correlation (r): {correlation:.4f}")
    print(f"R²: {correlation**2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Square Error: {rmse:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.4f}")
    print()
    
    print("Value Ranges:")
    print(f"y_true: [{df['y_true'].min():.2f}, {df['y_true'].max():.2f}]")
    print(f"y_pred: [{df['y_pred'].min():.2f}, {df['y_pred'].max():.2f}]")
    print()
    
    print("Sign Distribution:")
    true_pos_count = (df['y_true'] > 0).sum()
    true_neg_count = (df['y_true'] < 0).sum()
    pred_pos_count = (df['y_pred'] > 0).sum()
    pred_neg_count = (df['y_pred'] < 0).sum()
    
    print(f"y_true: {true_pos_count} positive, {true_neg_count} negative")
    print(f"y_pred: {pred_pos_count} positive, {pred_neg_count} negative")
    print("="*60)

def main():
    """Main function to create prediction plots."""
    parser = argparse.ArgumentParser(description='Plot y_true vs y_pred from TSV file')
    parser.add_argument('input_file', help='Input TSV file path')
    parser.add_argument('--output-dir', '-o', help='Directory to save plots (optional)')
    parser.add_argument('--no-show', action='store_true', help="Don't display plots, only save them")
    parser.add_argument('--correlation-only', '-c', action='store_true', help='Show only correlation plot')
    parser.add_argument('--sign-analysis', '-s', action='store_true', help='Show only sign analysis plots')
    parser.add_argument('--pairwise-diff', '-p', action='store_true', help='Show only pairwise difference plots')
    parser.add_argument('--all-plots', '-a', action='store_true', help='Generate all available plots')
    parser.add_argument('--max-lines', type=int, default=100, help='Maximum number of connecting lines to show (default: 100)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    df = load_data(args.input_file)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Validate required columns
    required_columns = ['y_true', 'y_pred']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"Data loaded successfully! Shape: {df.shape}")
    
    # Print summary statistics
    print_summary_stats(df)
    
    # Determine what to show
    show_plots = not args.no_show
    
    # Create plots based on arguments
    if args.correlation_only:
        create_correlation_plot(df, args.output_dir, show_plots)
    elif args.sign_analysis:
        create_sign_analysis_plot(df, args.output_dir, show_plots)
    elif args.pairwise_diff:
        create_pairwise_difference_plot(df, args.output_dir, show_plots, args.max_lines)
    elif args.all_plots:
        create_prediction_plots(df, args.output_dir, show_plots)
        create_correlation_plot(df, args.output_dir, show_plots)
        create_sign_analysis_plot(df, args.output_dir, show_plots)
        create_pairwise_difference_plot(df, args.output_dir, show_plots, args.max_lines)
    else:
        # Default: show pairwise difference plot (most useful for seeing connections)
        create_pairwise_difference_plot(df, args.output_dir, show_plots, args.max_lines)
    
    print("\nPlotting complete!")

if __name__ == "__main__":
    main()