#!/usr/bin/env python3
"""
Visualize pairwise melting temperature predictions with comprehensive statistical analysis.
Creates scatter plots comparing true vs predicted temperature differences for multiple tools.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from pathlib import Path

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def calculate_statistics(y_true, y_pred):
    """Calculate comprehensive statistics for predictions"""
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {}
    
    # Correlation metrics
    pearson_r, pearson_p = pearsonr(y_true_clean, y_pred_clean)
    spearman_r, spearman_p = spearmanr(y_true_clean, y_pred_clean)
    
    # Error metrics
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Additional metrics
    mean_error = np.mean(y_pred_clean - y_true_clean)  # Bias
    std_error = np.std(y_pred_clean - y_true_clean)    # Precision
    
    return {
        'n_pairs': len(y_true_clean),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mean_error': mean_error,
        'std_error': std_error
    }

def calculate_r80_correlation(df, min_pctid=0.8):
    """
    Calculate R80 correlation - correlation between actual and predicted absolute temperatures
    for pairs with sequence identity >= min_pctid, following the reference script logic.
    """
    # Filter by sequence identity threshold
    filtered_df = df[df['pctid'] >= min_pctid].copy()
    
    if len(filtered_df) == 0:
        return np.nan, 0
    
    # Calculate predicted absolute temperatures
    # tm1 is known, tm2 is predicted as tm1 + predicted_difference
    true_tm2 = filtered_df['tm2'].values
    pred_tm2 = filtered_df['tm1'].values + filtered_df['y_pred'].values
    
    # Calculate correlation using the same method as reference script
    def correl_reference(X, Y):
        n = len(X)
        if n == 0:
            return 0.0
        
        sumx = np.sum(X)
        sumy = np.sum(Y)
        sumx2 = np.sum(X * X)
        sumy2 = np.sum(Y * Y)
        sumxy = np.sum(X * Y)
        
        top = sumxy - (sumx * sumy) / n
        bottomL = sumx2 - (sumx * sumx) / n
        bottomR = sumy2 - (sumy * sumy) / n
        
        if bottomL <= 0 or bottomR <= 0:
            return 0.0
        
        return top / np.sqrt(bottomL * bottomR)
    
    r80 = correl_reference(true_tm2, pred_tm2)
    
    return r80, len(filtered_df)

def create_scatter_plot(df, tool_name, output_file, figsize=(10, 8), 
                       point_size=20, alpha=0.6, color_by_pctid=True):
    """Create scatter plot with statistical annotations"""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate statistics
    stats_dict = calculate_statistics(df['y_true'].values, df['y_pred'].values)
    
    # Calculate R80 if sequence identity is available
    r50, n_r50 = calculate_r80_correlation(df, min_pctid=0.5)
    stats_dict['r50'] = r50
    stats_dict['n_r50'] = n_r50
    
    # Create scatter plot
    if color_by_pctid and 'pctid' in df.columns:
        scatter = ax.scatter(df['y_true'], df['y_pred'], 
                           c=df['pctid'], cmap='viridis', 
                           s=point_size, alpha=alpha, edgecolors='white', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sequence Identity', fontsize=12)
    else:
        ax.scatter(df['y_true'], df['y_pred'], 
                  s=point_size, alpha=alpha, edgecolors='white', linewidth=0.5)
    
    # Add diagonal line (perfect prediction)
    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect prediction')
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(df['y_true'], df['y_pred'], 1)
        p = np.poly1d(z)
        ax.plot(df['y_true'], p(df['y_true']), 'b-', alpha=0.8, linewidth=2, label='Trend line')
    
    # Formatting
    ax.set_xlabel('True Temperature Difference (°C)', fontsize=14)
    ax.set_ylabel('Predicted Temperature Difference (°C)', fontsize=14)
    ax.set_title(f'{tool_name} - Pairwise Temperature Predictions', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add statistics text box
    r50_text = f", R50 = {stats_dict['r50']:.3f} (n={stats_dict['n_r50']})" if not np.isnan(stats_dict['r50']) else ""
    stats_text = f"""Statistics (n={stats_dict['n_pairs']}):
Pearson r = {stats_dict['pearson_r']:.3f} (p={stats_dict['pearson_p']:.2e})
Spearman ρ = {stats_dict['spearman_r']:.3f} (p={stats_dict['spearman_p']:.2e})
R² = {stats_dict['r2']:.3f}{r50_text}
RMSE = {stats_dict['rmse']:.2f} °C
MAE = {stats_dict['mae']:.2f} °C
Bias = {stats_dict['mean_error']:.2f} ± {stats_dict['std_error']:.2f} °C"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_dict

def create_comparison_plot(data_dict, output_file, figsize=(15, 10)):
    """Create multi-panel comparison plot for all tools"""
    
    n_tools = len(data_dict)
    if n_tools == 0:
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_tools)
    n_rows = (n_tools + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle different subplot configurations
    if n_tools == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        axes = list(axes)
    elif n_rows > 1 and n_cols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    stats_summary = {}
    
    for i, (tool_name, df) in enumerate(data_dict.items()):
        ax = axes[i]
        
        # Calculate statistics
        stats_dict = calculate_statistics(df['y_true'].values, df['y_pred'].values)
        
        # Calculate R80 if sequence identity is available
        r50, n_r50 = calculate_r80_correlation(df, min_pctid=0.5)
        stats_dict['r50'] = r50
        stats_dict['n_r50'] = n_r50
        
        stats_summary[tool_name] = stats_dict
        
        # Create scatter plot
        if 'pctid' in df.columns:
            scatter = ax.scatter(df['y_true'], df['y_pred'], 
                               c=df['pctid'], cmap='viridis', 
                               s=15, alpha=0.6, edgecolors='none')
        else:
            ax.scatter(df['y_true'], df['y_pred'], 
                      s=15, alpha=0.6, edgecolors='none')
        
        # Add diagonal line
        min_val = min(df['y_true'].min(), df['y_pred'].min())
        max_val = max(df['y_true'].max(), df['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['y_true'], df['y_pred'], 1)
            p = np.poly1d(z)
            ax.plot(df['y_true'], p(df['y_true']), 'b-', alpha=0.8, linewidth=1)
        
        # Formatting
        ax.set_xlabel('True ΔTm (°C)', fontsize=10)
        ax.set_ylabel('Predicted ΔTm (°C)', fontsize=10)
        r50_text = f", R50={stats_dict['r50']:.3f}" if not np.isnan(stats_dict['r50']) else ""
        ax.set_title(f'{tool_name}\nr={stats_dict["pearson_r"]:.3f}, RMSE={stats_dict["rmse"]:.1f}°C{r50_text}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_tools, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_summary

def create_performance_summary(stats_summary, output_file):
    """Create summary table and bar plots of performance metrics"""
    
    if not stats_summary:
        return
    
    # Convert to DataFrame
    df_stats = pd.DataFrame(stats_summary).T
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sort by Pearson correlation for consistent ordering
    df_stats_sorted = df_stats.sort_values('pearson_r', ascending=False)
    tools = df_stats_sorted.index
    
    # Plot 1: Correlation coefficients
    ax1 = axes[0, 0]
    x_pos = np.arange(len(tools))
    ax1.bar(x_pos - 0.2, df_stats_sorted['pearson_r'], 0.4, label='Pearson r', alpha=0.8)
    ax1.bar(x_pos + 0.2, df_stats_sorted['spearman_r'], 0.4, label='Spearman ρ', alpha=0.8)
    ax1.set_xlabel('Prediction Tool')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('Correlation Performance', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(tools, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Error metrics
    ax2 = axes[0, 1]
    ax2.bar(x_pos - 0.2, df_stats_sorted['rmse'], 0.4, label='RMSE', alpha=0.8)
    ax2.bar(x_pos + 0.2, df_stats_sorted['mae'], 0.4, label='MAE', alpha=0.8)
    ax2.set_xlabel('Prediction Tool')
    ax2.set_ylabel('Error (°C)')
    ax2.set_title('Error Metrics', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tools, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: R² and sample sizes
    ax3 = axes[1, 0]
    bars = ax3.bar(tools, df_stats_sorted['r2'], alpha=0.8)
    ax3.set_xlabel('Prediction Tool')
    ax3.set_ylabel('R² Score')
    ax3.set_title('Coefficient of Determination', fontweight='bold')
    ax3.set_xticks(range(len(tools)))
    ax3.set_xticklabels(tools, rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Add sample size annotations
    for i, (tool, r2) in enumerate(zip(tools, df_stats_sorted['r2'])):
        n_pairs = int(df_stats_sorted.loc[tool, 'n_pairs'])
        ax3.text(i, r2 + 0.02, f'n={n_pairs}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Bias analysis
    ax4 = axes[1, 1]
    ax4.errorbar(range(len(tools)), df_stats_sorted['mean_error'], 
                yerr=df_stats_sorted['std_error'], fmt='o', capsize=5, alpha=0.8)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Prediction Tool')
    ax4.set_ylabel('Bias ± Std (°C)')
    ax4.set_title('Prediction Bias', fontweight='bold')
    ax4.set_xticks(range(len(tools)))
    ax4.set_xticklabels(tools, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary table
    table_file = output_file.replace('.png', '_summary_table.tsv')
    df_stats.round(4).to_csv(table_file, sep='\t')
    print(f"Summary table saved to: {table_file}")
    
    return df_stats

def create_residuals_plot(data_dict, output_file, figsize=(15, 5)):
    """Create residuals analysis plots"""
    
    n_tools = len(data_dict)
    if n_tools == 0:
        return
    
    fig, axes = plt.subplots(1, n_tools, figsize=(5*n_tools, 5))
    if n_tools == 1:
        axes = [axes]
    
    for i, (tool_name, df) in enumerate(data_dict.items()):
        ax = axes[i]
        
        residuals = df['y_pred'] - df['y_true']
        
        # Residuals vs predicted
        ax.scatter(df['y_pred'], residuals, alpha=0.6, s=15)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('Predicted ΔTm (°C)')
        ax.set_ylabel('Residuals (°C)')
        ax.set_title(f'{tool_name} - Residuals Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add some statistics
        residual_std = np.std(residuals)
        ax.text(0.02, 0.98, f'Residual Std: {residual_std:.2f}°C', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize pairwise melting temperature predictions with statistical analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=        """
Examples:
    # Single tool visualization
    python visualize_pairwise_predictions.py --input pairwise_petai.tsv --output plots/petai_analysis.png --tool_name "PetAI"
    
    # Analyze all TSV files in directory
    python visualize_pairwise_predictions.py --input_dir pairwise_data/ --output_dir plots/
    
    # Custom analysis
    python visualize_pairwise_predictions.py --input pairwise_petai.tsv --output plots/petai.png --tool_name "PetAI" --figsize 12 10 --point_size 30
        """
    )
    
    # Input options
    parser.add_argument("--input", "-i", type=str, 
                        help="Single pairwise TSV file to analyze")
    parser.add_argument("--input_dir", type=str,
                        help="Directory containing TSV files with pairwise data")
    
    # Output options
    parser.add_argument("--output", "-o", type=str,
                        help="Output file for single tool analysis")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Output directory for plots (default: plots)")
    
    # Analysis options
    parser.add_argument("--tool_name", type=str,
                        help="Name of the prediction tool (for single file)")
    parser.add_argument("--min_pairs", type=int, default=10,
                        help="Minimum number of pairs required for analysis (default: 10)")
    
    # Plot customization
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 8],
                        help="Figure size (width height) (default: 10 8)")
    parser.add_argument("--point_size", type=int, default=20,
                        help="Scatter plot point size (default: 20)")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Point transparency (default: 0.6)")
    parser.add_argument("--no_color_by_pctid", action="store_true",
                        help="Don't color points by sequence identity")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Single file analysis
    if args.input:
        if not args.output:
            tool_name = args.tool_name or Path(args.input).stem
            args.output = os.path.join(args.output_dir, f"{tool_name}_scatter.png")
        
        print(f"Analyzing {args.input}...")
        df = pd.read_csv(args.input, sep='\t')
        tool_name = args.tool_name or Path(args.input).stem
        
        stats_dict = create_scatter_plot(
            df, tool_name, args.output, 
            figsize=tuple(args.figsize),
            point_size=args.point_size,
            alpha=args.alpha,
            color_by_pctid=not args.no_color_by_pctid
        )
        
        print(f"Plot saved to: {args.output}")
        print(f"Statistics: r={stats_dict['pearson_r']:.3f}, RMSE={stats_dict['rmse']:.2f}°C")
    
    # Multiple files comparison
    elif args.input_dir:
        print(f"Analyzing all TSV files in {args.input_dir}...")
        
        # Find all TSV files in the directory
        data_dict = {}
        tsv_files = list(Path(args.input_dir).glob("*.tsv"))
        
        if not tsv_files:
            print("No TSV files found in input directory")
            return 1
        
        for file_path in tsv_files:
            tool_name = file_path.stem
            # Remove common prefixes if they exist
            if tool_name.startswith("pairwise_"):
                tool_name = tool_name.replace("pairwise_", "")
            
            try:
                df = pd.read_csv(file_path, sep='\t')
                
                # Validate required columns
                required_cols = ['y_true', 'y_pred']
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Skipping {file_path.name} - missing required columns {required_cols}")
                    print(f"  Available columns: {list(df.columns)}")
                    continue
                
                data_dict[tool_name] = df
                print(f"Loaded {len(df)} pairs for {tool_name}")
                
            except Exception as e:
                print(f"Warning: Could not load {file_path.name}: {e}")
                continue
        
        print(f"\nAnalyzing {len(data_dict)} tools with sufficient data...")
        
        # Create individual plots
        for tool_name, df in data_dict.items():
            output_file = os.path.join(args.output_dir, f"{tool_name}_scatter.png")
            create_scatter_plot(df, tool_name, output_file, 
                              figsize=tuple(args.figsize),
                              point_size=args.point_size,
                              alpha=args.alpha,
                              color_by_pctid=not args.no_color_by_pctid)
        
        # Create comparison plot
        comparison_file = os.path.join(args.output_dir, "tools_comparison.png")
        stats_summary = create_comparison_plot(data_dict, comparison_file)
        
        # Create performance summary
        summary_file = os.path.join(args.output_dir, "performance_summary.png")
        create_performance_summary(stats_summary, summary_file)
        
        # Create residuals analysis
        residuals_file = os.path.join(args.output_dir, "residuals_analysis.png")
        create_residuals_plot(data_dict, residuals_file)
        
        print(f"\nAll plots saved to: {args.output_dir}/")
        print("Generated files:")
        print(f"- Individual scatter plots: {{tool}}_scatter.png ({len(data_dict)} files)")
        print("- Tools comparison: tools_comparison.png")
        print("- Performance summary: performance_summary.png")
        print("- Residuals analysis: residuals_analysis.png")
        print("- Summary table: performance_summary_summary_table.tsv")
        
        # Print quick summary
        print(f"\nQuick Performance Summary:")
        df_stats = pd.DataFrame(stats_summary).T
        df_stats_sorted = df_stats.sort_values('pearson_r', ascending=False)
        for tool in df_stats_sorted.index:
            r = df_stats_sorted.loc[tool, 'pearson_r']
            rmse = df_stats_sorted.loc[tool, 'rmse']
            n = int(df_stats_sorted.loc[tool, 'n_pairs'])
            r50 = df_stats_sorted.loc[tool, 'r50']
            r50_text = f", R50={r50:.3f}" if not np.isnan(r50) else ""
            print(f"  {tool:12s}: r={r:6.3f}, RMSE={rmse:6.2f}°C, n={n:4d}{r50_text}")
    
    else:
        print("Error: Must specify either --input or --input_dir")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())