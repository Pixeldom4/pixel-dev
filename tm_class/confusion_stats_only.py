import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse
import sys
import os
import glob
from pathlib import Path
from datetime import datetime

def load_data(file_path):
    """Load TSV data from file."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def classify_prediction(y_true, y_pred):
    """Classify prediction into TP, FP, TN, FN based on sign."""
    actual_positive = y_true > 0
    predicted_positive = y_pred > 0
    
    if actual_positive and predicted_positive:
        return 'TP'
    elif not actual_positive and predicted_positive:
        return 'FP'
    elif actual_positive and not predicted_positive:
        return 'FN'
    else:
        return 'TN'

def calculate_metrics(df):
    """Calculate performance metrics and return as dictionary."""
    # Convert to binary classifications
    y_true_binary = (df['y_true'] > 0).astype(int)
    y_pred_binary = (df['y_pred'] > 0).astype(int)
    
    # Count classifications
    classifications = df.apply(lambda row: classify_prediction(row['y_true'], row['y_pred']), axis=1)
    tp_count = (classifications == 'TP').sum()
    fp_count = (classifications == 'FP').sum()
    fn_count = (classifications == 'FN').sum()
    tn_count = (classifications == 'TN').sum()
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Calculate additional metrics
    total = len(df)
    specificity = tn_count / (tn_count + fp_count) if (tn_count + fp_count) > 0 else 0
    sensitivity = recall  # Same as recall
    
    # Calculate directional accuracy (same sign)
    same_sign = ((df['y_true'] > 0) & (df['y_pred'] > 0)) | ((df['y_true'] < 0) & (df['y_pred'] < 0))
    directional_accuracy = same_sign.sum() / len(df)
    
    # Calculate correlation
    correlation = df['y_true'].corr(df['y_pred'])
    
    # Calculate MAE and RMSE for regression metrics
    mae = np.mean(np.abs(df['y_true'] - df['y_pred']))
    rmse = np.sqrt(np.mean((df['y_true'] - df['y_pred'])**2))
    
    # Bias analysis
    bias = df['y_pred'].mean() - df['y_true'].mean()
    
    # Sign distribution
    true_positive_count = (df['y_true'] > 0).sum()
    true_negative_count = (df['y_true'] < 0).sum()
    pred_positive_count = (df['y_pred'] > 0).sum()
    pred_negative_count = (df['y_pred'] < 0).sum()
    
    return {
        'tp_count': tp_count,
        'fp_count': fp_count,
        'fn_count': fn_count,
        'tn_count': tn_count,
        'total': total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'directional_accuracy': directional_accuracy,
        'correlation': correlation,
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'y_true_mean': df['y_true'].mean(),
        'y_true_std': df['y_true'].std(),
        'y_true_min': df['y_true'].min(),
        'y_true_max': df['y_true'].max(),
        'y_pred_mean': df['y_pred'].mean(),
        'y_pred_std': df['y_pred'].std(),
        'y_pred_min': df['y_pred'].min(),
        'y_pred_max': df['y_pred'].max(),
        'true_positive_count': true_positive_count,
        'true_negative_count': true_negative_count,
        'pred_positive_count': pred_positive_count,
        'pred_negative_count': pred_negative_count
    }

def print_confusion_matrix_stats(metrics, title="CONFUSION MATRIX STATISTICS"):
    """Print just the confusion matrix statistics in a compact format."""
    print("CONFUSION MATRIX STATISTICS")
    print("-" * 30)
    print(f"True Positives (TP):  {metrics['tp_count']}")
    print(f"False Positives (FP): {metrics['fp_count']}")
    print(f"False Negatives (FN): {metrics['fn_count']}")
    print(f"True Negatives (TN):  {metrics['tn_count']}")
    print("-" * 30)
    print(f"Total samples: {metrics['total']}")
    print(f"Positive samples: {metrics['tp_count'] + metrics['fn_count']}")
    print(f"Negative samples: {metrics['tn_count'] + metrics['fp_count']}")
    
    print("CLASSIFICATION METRICS")
    print("-" * 30)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    
    print("DIRECTIONAL METRICS")
    print("-" * 30)
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.4f}")
    print(f"Correlation (r):      {metrics['correlation']:.4f}")
    print()

def print_detailed_metrics(metrics, title="PERFORMANCE METRICS"):
    """Print detailed metrics in a formatted way."""
    print("=" * 50)
    print(title)
    print("=" * 50)
    print("CONFUSION MATRIX STATISTICS")
    print("-" * 30)
    print(f"True Positives (TP):  {metrics['tp_count']}")
    print(f"False Positives (FP): {metrics['fp_count']}")
    print(f"False Negatives (FN): {metrics['fn_count']}")
    print(f"True Negatives (TN):  {metrics['tn_count']}")
    print("-" * 30)
    print(f"Total samples: {metrics['total']}")
    print(f"Positive samples: {metrics['tp_count'] + metrics['fn_count']}")
    print(f"Negative samples: {metrics['tn_count'] + metrics['fp_count']}")
    print()
    
    print("CLASSIFICATION METRICS")
    print("-" * 30)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print()
    
    print("DIRECTIONAL METRICS")
    print("-" * 30)
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.4f}")
    print(f"Correlation (r):      {metrics['correlation']:.4f}")
    print()
    
    print("REGRESSION METRICS")
    print("-" * 30)
    print(f"Mean Absolute Error:  {metrics['mae']:.4f}")
    print(f"Root Mean Sq Error:   {metrics['rmse']:.4f}")
    print()
    
    # Distribution statistics
    print("DISTRIBUTION STATISTICS")
    print("-" * 30)
    print("y_true distribution:")
    print(f"  Mean: {metrics['y_true_mean']:.4f}")
    print(f"  Std:  {metrics['y_true_std']:.4f}")
    print(f"  Min:  {metrics['y_true_min']:.4f}")
    print(f"  Max:  {metrics['y_true_max']:.4f}")
    print()
    print("y_pred distribution:")
    print(f"  Mean: {metrics['y_pred_mean']:.4f}")
    print(f"  Std:  {metrics['y_pred_std']:.4f}")
    print(f"  Min:  {metrics['y_pred_min']:.4f}")
    print(f"  Max:  {metrics['y_pred_max']:.4f}")
    print()
    
    # Sign distribution
    print("SIGN DISTRIBUTION")
    print("-" * 30)
    print(f"y_true: {metrics['true_positive_count']} positive, {metrics['true_negative_count']} negative")
    print(f"y_pred: {metrics['pred_positive_count']} positive, {metrics['pred_negative_count']} negative")
    print()
    
    # Bias analysis
    print("BIAS ANALYSIS")
    print("-" * 30)
    print(f"Prediction bias: {metrics['bias']:.4f}")
    if metrics['bias'] > 0:
        print("  Model tends to over-predict")
    elif metrics['bias'] < 0:
        print("  Model tends to under-predict")
    else:
        print("  Model is unbiased on average")
    
    print("=" * 50)

def get_tsv_files(input_path):
    """Get list of TSV files from input path (file or directory)."""
    if os.path.isfile(input_path):
        if input_path.endswith('.tsv'):
            return [input_path]
        else:
            print(f"Warning: {input_path} is not a TSV file")
            return []
    elif os.path.isdir(input_path):
        # Find all TSV files in directory
        tsv_files = glob.glob(os.path.join(input_path, "*.tsv"))
        if not tsv_files:
            print(f"No TSV files found in directory: {input_path}")
        return sorted(tsv_files)
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        return []

def save_metrics_to_csv(file_metrics_list, output_path=None):
    """Save metrics for all files to a CSV file."""
    if not file_metrics_list:
        print("No metrics to save to CSV")
        return
    
    # Prepare data for CSV
    csv_data = []
    for file_path, metrics in file_metrics_list:
        row = {
            'filename': os.path.basename(file_path),
            'filepath': file_path,
            'total_samples': metrics['total'],
            'tp': metrics['tp_count'],
            'fp': metrics['fp_count'],
            'fn': metrics['fn_count'],
            'tn': metrics['tn_count'],
            'positive_samples': metrics['tp_count'] + metrics['fn_count'],
            'negative_samples': metrics['tn_count'] + metrics['fp_count'],
            'accuracy': round(metrics['accuracy'], 4),
            'precision': round(metrics['precision'], 4),
            'recall': round(metrics['recall'], 4),
            'specificity': round(metrics['specificity'], 4),
            'f1_score': round(metrics['f1'], 4),
            'directional_accuracy': round(metrics['directional_accuracy'], 4),
            'correlation': round(metrics['correlation'], 4),
            'mae': round(metrics['mae'], 4),
            'rmse': round(metrics['rmse'], 4),
            'bias': round(metrics['bias'], 4),
            'y_true_mean': round(metrics['y_true_mean'], 4),
            'y_true_std': round(metrics['y_true_std'], 4),
            'y_pred_mean': round(metrics['y_pred_mean'], 4),
            'y_pred_std': round(metrics['y_pred_std'], 4)
        }
        csv_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    
    # Generate output filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"confusion_matrix_stats_{timestamp}.csv"
    
    # Save to CSV
    try:
        df.to_csv(output_path, index=False)
        print(f"\nMetrics saved to: {output_path}")
        print(f"Saved statistics for {len(csv_data)} files")
        
        # Show column info
        print(f"\nCSV contains the following columns:")
        print(f"  Basic info: filename, filepath, total_samples")
        print(f"  Confusion matrix: tp, fp, fn, tn, positive_samples, negative_samples")
        print(f"  Classification: accuracy, precision, recall, specificity, f1_score")
        print(f"  Other metrics: directional_accuracy, correlation, mae, rmse, bias")
        print(f"  Distributions: y_true_mean, y_true_std, y_pred_mean, y_pred_std")
        
    except Exception as e:
        print(f"Error saving CSV: {e}")

def add_aggregate_row_to_csv(csv_path, aggregate_metrics, num_files):
    """Add aggregate statistics as the last row in the CSV."""
    try:
        # Read existing CSV
        df = pd.read_csv(csv_path)
        
        # Create aggregate row
        aggregate_row = {
            'filename': f'AGGREGATE_{num_files}_files',
            'filepath': 'AGGREGATE',
            'total_samples': aggregate_metrics['total'],
            'tp': aggregate_metrics['tp_count'],
            'fp': aggregate_metrics['fp_count'],
            'fn': aggregate_metrics['fn_count'],
            'tn': aggregate_metrics['tn_count'],
            'positive_samples': aggregate_metrics['tp_count'] + aggregate_metrics['fn_count'],
            'negative_samples': aggregate_metrics['tn_count'] + aggregate_metrics['fp_count'],
            'accuracy': round(aggregate_metrics['accuracy'], 4),
            'precision': round(aggregate_metrics['precision'], 4),
            'recall': round(aggregate_metrics['recall'], 4),
            'specificity': round(aggregate_metrics['specificity'], 4),
            'f1_score': round(aggregate_metrics['f1'], 4),
            'directional_accuracy': round(aggregate_metrics['directional_accuracy'], 4),
            'correlation': round(aggregate_metrics['correlation'], 4),
            'mae': round(aggregate_metrics['mae'], 4),
            'rmse': round(aggregate_metrics['rmse'], 4),
            'bias': round(aggregate_metrics['bias'], 4),
            'y_true_mean': round(aggregate_metrics['y_true_mean'], 4),
            'y_true_std': round(aggregate_metrics['y_true_std'], 4),
            'y_pred_mean': round(aggregate_metrics['y_pred_mean'], 4),
            'y_pred_std': round(aggregate_metrics['y_pred_std'], 4)
        }
        
        # Add aggregate row
        df = pd.concat([df, pd.DataFrame([aggregate_row])], ignore_index=True)
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        print(f"Added aggregate statistics to CSV")
        
    except Exception as e:
        print(f"Error adding aggregate row to CSV: {e}")
    

def aggregate_metrics(all_metrics):
    """Aggregate metrics across multiple files."""
    if not all_metrics:
        return None
    
    # Sum up counts
    total_tp = sum(m['tp_count'] for m in all_metrics)
    total_fp = sum(m['fp_count'] for m in all_metrics)
    total_fn = sum(m['fn_count'] for m in all_metrics)
    total_tn = sum(m['tn_count'] for m in all_metrics)
    total_samples = sum(m['total'] for m in all_metrics)
    
    # Calculate aggregate metrics
    accuracy = (total_tp + total_tn) / total_samples
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Weighted averages for other metrics
    weights = [m['total'] for m in all_metrics]
    total_weight = sum(weights)
    
    directional_accuracy = sum(m['directional_accuracy'] * w for m, w in zip(all_metrics, weights)) / total_weight
    correlation = sum(m['correlation'] * w for m, w in zip(all_metrics, weights)) / total_weight
    mae = sum(m['mae'] * w for m, w in zip(all_metrics, weights)) / total_weight
    rmse = np.sqrt(sum(m['rmse']**2 * w for m, w in zip(all_metrics, weights)) / total_weight)
    bias = sum(m['bias'] * w for m, w in zip(all_metrics, weights)) / total_weight
    
    return {
        'tp_count': total_tp,
        'fp_count': total_fp,
        'fn_count': total_tn,
        'tn_count': total_tn,
        'total': total_samples,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'directional_accuracy': directional_accuracy,
        'correlation': correlation,
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'y_true_mean': sum(m['y_true_mean'] * w for m, w in zip(all_metrics, weights)) / total_weight,
        'y_true_std': np.sqrt(sum(m['y_true_std']**2 * w for m, w in zip(all_metrics, weights)) / total_weight),
        'y_true_min': min(m['y_true_min'] for m in all_metrics),
        'y_true_max': max(m['y_true_max'] for m in all_metrics),
        'y_pred_mean': sum(m['y_pred_mean'] * w for m, w in zip(all_metrics, weights)) / total_weight,
        'y_pred_std': np.sqrt(sum(m['y_pred_std']**2 * w for m, w in zip(all_metrics, weights)) / total_weight),
        'y_pred_min': min(m['y_pred_min'] for m in all_metrics),
        'y_pred_max': max(m['y_pred_max'] for m in all_metrics),
        'true_positive_count': sum(m['true_positive_count'] for m in all_metrics),
        'true_negative_count': sum(m['true_negative_count'] for m in all_metrics),
        'pred_positive_count': sum(m['pred_positive_count'] for m in all_metrics),
        'pred_negative_count': sum(m['pred_negative_count'] for m in all_metrics)
    }


def main():
    """Main function to calculate confusion matrix statistics."""
    parser = argparse.ArgumentParser(description='Calculate confusion matrix and performance statistics')
    parser.add_argument('input_path', help='Input TSV file path or directory containing TSV files')
    parser.add_argument('--show-data', '-s', action='store_true', help='Show first few rows of data')
    parser.add_argument('--individual', '-i', action='store_true', help='Show individual file results when processing directory')
    parser.add_argument('--summary-only', '-so', action='store_true', help='Show only summary when processing directory')
    parser.add_argument('--confusion-only', '-co', action='store_true', help='Show only confusion matrix stats for each file')
    parser.add_argument('--csv-output', '-csv', type=str, help='Save metrics to CSV file (specify filename or use auto-generated name)')
    parser.add_argument('--csv-only', action='store_true', help='Save to CSV without console output (except errors)')
    
    args = parser.parse_args()
    
    # Get TSV files
    tsv_files = get_tsv_files(args.input_path)
    
    if not tsv_files:
        sys.exit(1)
    
    print(f"Found {len(tsv_files)} TSV file(s) to process")
    
    all_metrics = []
    successful_files = []
    file_metrics_list = []  # For CSV output
    
    for file_path in tsv_files:
        if not args.csv_only:
            print(f"\nProcessing: {file_path}")
        
        # Load data
        df = load_data(file_path)
        if df is None:
            if not args.csv_only:
                print(f"Skipping {file_path} due to loading error")
            continue
        
        # Validate required columns
        required_columns = ['y_true', 'y_pred']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            if not args.csv_only:
                print(f"Error: Missing required columns in {file_path}: {missing_columns}")
                print(f"Available columns: {list(df.columns)}")
            continue
        
        if not args.csv_only:
            print(f"Data loaded successfully! Shape: {df.shape}")
        
        if args.show_data and not args.csv_only:
            print("\nFirst 5 rows of data:")
            print(df[['y_true', 'y_pred']].head())
            print()
        
        # Calculate metrics
        metrics = calculate_metrics(df)
        all_metrics.append(metrics)
        successful_files.append(file_path)
        file_metrics_list.append((file_path, metrics))
        
        # Show individual results based on options (skip if csv_only)
        if not args.csv_only:
            if args.confusion_only:
                print(f"\n--- Results for {os.path.basename(file_path)} ---")
                print_confusion_matrix_stats(metrics)
            elif args.individual or (len(tsv_files) == 1 and not args.summary_only):
                print_detailed_metrics(metrics, f"RESULTS FOR {os.path.basename(file_path)}")
            elif len(tsv_files) > 1 and not args.summary_only:
                # For multiple files, show compact stats by default
                print(f"\n--- Results for {os.path.basename(file_path)} ---")
                print_confusion_matrix_stats(metrics)
    
    # Save to CSV if requested
    csv_path = None
    if args.csv_output is not None or args.csv_only:
        if args.csv_output and args.csv_output.strip():
            csv_path = args.csv_output
        else:
            csv_path = None  # Will auto-generate filename
        
        save_metrics_to_csv(file_metrics_list, csv_path)
        if csv_path is None:
            # Get the auto-generated filename for aggregate row
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"confusion_matrix_stats_{timestamp}.csv"
    
    # Show aggregate results if multiple files and not csv_only
    if len(successful_files) > 1 and not args.summary_only and not args.csv_only:
        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS ACROSS ALL FILES")
        print(f"{'='*60}")
        print(f"Successfully processed {len(successful_files)} files:")
        for f in successful_files:
            print(f"  - {os.path.basename(f)}")
        print()
        
        aggregate_metrics_result = aggregate_metrics(all_metrics)
        if aggregate_metrics_result:
            print_detailed_metrics(aggregate_metrics_result, "AGGREGATE PERFORMANCE METRICS")
            
            # Add aggregate row to CSV if CSV output was requested
            if csv_path and len(successful_files) > 1:
                add_aggregate_row_to_csv(csv_path, aggregate_metrics_result, len(successful_files))
    
    if len(successful_files) == 0:
        print("No files were successfully processed.")
        sys.exit(1)
    
    if not args.csv_only:
        print(f"\nProcessing complete! Successfully analyzed {len(successful_files)} file(s).")

if __name__ == "__main__":
    main()