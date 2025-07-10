import pandas as pd
import matplotlib.pyplot as plt
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

def plot_y_pred_sign(df, target_label, output_dir=None, show_plots=True):
    """Plot sequences by Tm values colored by prediction accuracy (over/underestimate) for a chosen label."""
    
    # Filter data for the target label in either label1 or label2
    label_data = df[(df['label1'] == target_label) | (df['label2'] == target_label)].copy()
    
    if len(label_data) == 0:
        print(f"No data found for label: {target_label}")
        return None
    
    # Reset index to maintain original TSV order
    label_data = label_data.reset_index(drop=True)
    
    # Determine which Tm value corresponds to the OTHER sequence (not the target label)
    other_tm_values = []
    for _, row in label_data.iterrows():
        if row['label1'] == target_label:
            other_tm_values.append(row['tm2_true'])  # Plot tm2 when target is label1
        else:  # row['label2'] == target_label
            other_tm_values.append(row['tm1_true'])  # Plot tm1 when target is label2
    
    # Create color mapping based on y_pred vs y_true
    colors = ['red' if pred < true else 'blue' for pred, true in zip(label_data['y_pred'], label_data['y_true'])]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot sequences by Tm values
    x_positions = range(len(label_data))
    plt.scatter(x_positions, other_tm_values, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add color legend
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='y_pred < y_true (underestimate)')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='y_pred > y_true (overestimate)')
    
    plt.xlabel('Sequence Order (TSV row order)')
    plt.ylabel(f'Tm Value of Paired Sequence (with {target_label})')
    plt.title(f'Paired Sequence Tm Values by Order for {target_label}')
    plt.legend(handles=[red_patch, blue_patch])
    plt.grid(True, alpha=0.3)
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_label = target_label.replace('/', '_').replace('\\', '_')
        filename = f'tm_sequence_plot_{safe_label}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(output_dir, filename)}")
    
    # Show plot if requested
    if show_plots:
        plt.show()
    
    return label_data

def list_available_labels(df):
    """List all available labels from both label1 and label2 columns."""
    all_labels = pd.concat([df['label1'], df['label2']]).unique()
    
    label_info = []
    for label in all_labels:
        label1_count = sum(df['label1'] == label)
        label2_count = sum(df['label2'] == label)
        total_count = label1_count + label2_count
        
        label_info.append({
            'label': label,
            'total_appearances': total_count,
            'as_label1': label1_count,
            'as_label2': label2_count
        })
    
    # Sort by total appearances
    label_info.sort(key=lambda x: x['total_appearances'], reverse=True)
    
    print("Available labels:")
    print("=" * 60)
    print(f"{'Label':<20} {'Total':<8} {'As label1':<12} {'As label2':<12}")
    print("-" * 60)
    
    for info in label_info:
        print(f"{info['label']:<20} {info['total_appearances']:<8} {info['as_label1']:<12} {info['as_label2']:<12}")
    
    print(f"\nTotal unique labels: {len(all_labels)}")
    return label_info

def main():
    parser = argparse.ArgumentParser(description='Plot sequences by Tm values colored by over/underestimate')
    parser.add_argument('input_file', help='Input TSV file path')
    parser.add_argument('--label', '-l', type=str, help='Target label to plot')
    parser.add_argument('--list-labels', action='store_true', help='List all available labels and exit')
    parser.add_argument('--output-dir', '-o', help='Directory to save plots (optional)')
    parser.add_argument('--no-show', action='store_true', help="Don't display plots, only save them")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    df = load_data(args.input_file)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Validate required columns
    required_columns = ['y_true', 'y_pred', 'label1', 'label2']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"Data loaded successfully! Shape: {df.shape}")
    
    # List labels if requested
    if args.list_labels:
        list_available_labels(df)
        return
    
    # Check if label is provided
    if not args.label:
        print("Error: --label is required. Use --list-labels to see available options.")
        return
    
    # Check if the label exists
    if args.label not in df['label1'].values and args.label not in df['label2'].values:
        print(f"Error: label '{args.label}' not found in either label1 or label2 columns.")
        print("\nUse --list-labels to see all available labels.")
        return
    
    # Create plot
    show_plots = not args.no_show
    label_data = plot_y_pred_sign(df, args.label, args.output_dir, show_plots)
    
    if label_data is not None:
        underestimate_count = sum(1 for pred, true in zip(label_data['y_pred'], label_data['y_true']) if pred < true)
        overestimate_count = sum(1 for pred, true in zip(label_data['y_pred'], label_data['y_true']) if pred > true)
        exact_count = len(label_data) - underestimate_count - overestimate_count
        print(f"\nAnalysis complete for {args.label}!")
        print(f"Processed {len(label_data)} sequences.")
        print(f"Underestimates (y_pred < y_true): {underestimate_count} ({underestimate_count/len(label_data)*100:.1f}%)")
        print(f"Overestimates (y_pred > y_true): {overestimate_count} ({overestimate_count/len(label_data)*100:.1f}%)")
        print(f"Exact predictions: {exact_count} ({exact_count/len(label_data)*100:.1f}%)")

if __name__ == "__main__":
    main()