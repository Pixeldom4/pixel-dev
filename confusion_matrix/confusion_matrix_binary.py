import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import sys

def load_data(file_path):
    """Load TSV data from file."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def classify_prediction(y_true, y_pred):
    """Classify based on y_true sign only."""
    actual_positive = y_true > 0
    
    if actual_positive:
        return 'POS'  # Positive relationship
    else:
        return 'NEG'  # Negative relationship

def create_confusion_matrix_data(df, mirror_diagonal=True):  # Changed default to True
    """Create matrix data structure for visualization."""
    # Extract sequence identifiers
    df['label1_clean'] = df['label1'].astype(str).str.replace('tensor(', '').str.replace(')', '')
    df['label2_clean'] = df['label2'].astype(str)
    
    # Get unique sequences in order of appearance (preserve input order)
    seen = set()
    all_sequences = []
    for seq in list(df['label1_clean']) + list(df['label2_clean']):
        if seq not in seen:
            all_sequences.append(seq)
            seen.add(seq)
    
    # Create empty matrix
    matrix_data = pd.DataFrame(index=all_sequences, columns=all_sequences, dtype=object)
    matrix_values = pd.DataFrame(index=all_sequences, columns=all_sequences, dtype=float)
    
    # Fill matrix with classifications
    for _, row in df.iterrows():
        ref_seq = row['label1_clean']
        target_seq = row['label2_clean']
        classification = classify_prediction(row['y_true'], row['y_pred'])
        
        matrix_data.loc[ref_seq, target_seq] = classification
        matrix_values.loc[ref_seq, target_seq] = row['y_true']  # Use actual y_true value
        
        # Mirror across diagonal if requested
        if mirror_diagonal and ref_seq != target_seq:
            # Mirror with flipped classification and negated values
            flipped_classification = 'NEG' if classification == 'POS' else 'POS'
            matrix_data.loc[target_seq, ref_seq] = flipped_classification
            matrix_values.loc[target_seq, ref_seq] = -row['y_true']  # Negate the value
    
    return matrix_data, matrix_values, all_sequences

def plot_confusion_matrix(matrix_data, matrix_values, all_sequences, output_file=None, mirror_diagonal=True):  # Changed default to True
    """Create and display the confusion matrix visualization."""
    # Create numeric matrix for plotting
    classification_map = {'POS': 1, 'NEG': 2}
    numeric_matrix = matrix_data.applymap(lambda x: classification_map.get(x, 0))
    
    # Set up the plot
    plt.figure(figsize=(max(12, len(all_sequences) * 0.8), max(8, len(all_sequences) * 0.6)))
    
    # Create custom colormap
    colors = ['white', '#2ecc71', '#e74c3c']  # white, green (positive), red (negative)
    n_bins = 3
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Create the heatmap
    ax = sns.heatmap(numeric_matrix, 
                     annot=matrix_data, 
                     fmt='', 
                     cmap=cmap, 
                     cbar=False,
                     square=True,
                     linewidths=0.5,
                     linecolor='gray',
                     xticklabels=all_sequences,
                     yticklabels=all_sequences)
    
    # Add diagonal line if mirroring is enabled
    if mirror_diagonal:
        ax.plot([0, len(all_sequences)], [0, len(all_sequences)], 'k--', alpha=0.5, linewidth=2)
        title_suffix = " (with diagonal mirroring - flipped values)"
    else:
        title_suffix = ""
    
    # Customize the plot
    plt.title(f'Sequence Pair Relationship Matrix{title_suffix}\n(Reference sequences vs Target sequences)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Target Sequences', fontsize=12, fontweight='bold')
    plt.ylabel('Reference Sequences', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#2ecc71', label='Positive Relationship (POS)'),
        plt.Rectangle((0,0),1,1, facecolor='#e74c3c', label='Negative Relationship (NEG)')
    ]
    if mirror_diagonal:
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5, label='Diagonal (mirror line)'))
    
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_file}")
    
    plt.show()

def calculate_metrics(df):
    """Calculate and display basic statistics."""
    # Count classifications based on y_true only
    classifications = df.apply(lambda row: classify_prediction(row['y_true'], row['y_pred']), axis=1)
    pos_count = (classifications == 'POS').sum()
    neg_count = (classifications == 'NEG').sum()
    
    # Calculate prediction accuracy for reference
    y_true_binary = (df['y_true'] > 0).astype(int)
    y_pred_binary = (df['y_pred'] > 0).astype(int)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    # Print results
    print("\n" + "="*50)
    print("SEQUENCE RELATIONSHIP STATISTICS")
    print("="*50)
    print(f"Positive Relationships:  {pos_count}")
    print(f"Negative Relationships:  {neg_count}")
    print(f"Total Pairs:            {len(df)}")
    print("-"*30)
    print(f"Prediction Accuracy:    {accuracy:.3f}")
    print("="*50)

def main():
    """Main function to run the confusion matrix generator."""
    parser = argparse.ArgumentParser(description='Generate sequence pair relationship matrix')
    parser.add_argument('input_file', help='Input TSV file path')
    parser.add_argument('--output', '-o', help='Output image file path (optional)')
    parser.add_argument('--show-data', '-s', action='store_true', help='Show first few rows of data')
    parser.add_argument('--no-mirror', '-n', action='store_true', help='Disable diagonal mirroring with flipped values (mirroring is enabled by default)')  # Changed flag logic
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    df = load_data(args.input_file)
    
    # Validate required columns
    required_columns = ['y_true', 'y_pred', 'label1', 'label2']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Data loaded successfully! Shape: {df.shape}")
    
    if args.show_data:
        print("\nFirst 5 rows of data:")
        print(df.head())
    
    # Reverse the mirroring logic - now enabled by default, disabled with --no-mirror
    mirror_enabled = not args.no_mirror
    
    if mirror_enabled:
        print("Mirror mode enabled by default: Matrix will be mirrored across the diagonal with flipped values")
    else:
        print("Mirror mode disabled: Matrix will not be mirrored across the diagonal")
    
    # Create confusion matrix data
    print("\nGenerating relationship matrix...")
    matrix_data, matrix_values, all_sequences = create_confusion_matrix_data(df, mirror_enabled)
    
    # Calculate and display metrics
    calculate_metrics(df)
    
    # Create visualization
    print("\nCreating visualization...")
    plot_confusion_matrix(matrix_data, matrix_values, all_sequences, args.output, mirror_enabled)

def demo_with_sample_data():
    """Demo function with sample data for testing."""
    sample_data = {
        'y_true': [9.17, 26.61, 2.24, 20.95, 1.29, 1.22, -12.06, 10.25, 4.32, -0.39, 4.9, 3.97, 3.06, 2.39],
        'y_pred': [-11.9, -5.62, 0.88, 7.96, 0.55, -4.51, -3.95, -2.00, 1.65, 0.93, 1.58, 2.12, 1.58, 1.88],
        'pctid': [0.1457, 0.2053, 0.2185, 0.1921, 0.1921, 0.1921, 0.2185, 0.1987, 0.2119, 0.2053, 0.2119, 0.2119, 0.2119, 0.2119],
        'label1': ['tensor(102)']*14,
        'label2': [202, 306, 407, 501, 504, 601, 606, 611, 701, 702, 703, 704, 705, 706],
        'tm1': [65.96]*14,
        'tm2': [75.13, 92.57, 68.2, 86.91, 67.25, 67.18, 53.9, 76.21, 70.28, 65.57, 70.86, 69.93, 69.02, 68.35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Running demo with sample data...")
    print(f"Sample data shape: {df.shape}")
    
    # Show both versions - now mirrored by default first
    print("\n1. Mirrored matrix with flipped values (default behavior):")
    matrix_data_mirror, matrix_values_mirror, all_sequences_mirror = create_confusion_matrix_data(df, mirror_diagonal=True)
    calculate_metrics(df)
    plot_confusion_matrix(matrix_data_mirror, matrix_values_mirror, all_sequences_mirror, mirror_diagonal=True)
    
    print("\n2. Standard matrix (no mirroring):")
    matrix_data, matrix_values, all_sequences = create_confusion_matrix_data(df, mirror_diagonal=False)
    plot_confusion_matrix(matrix_data, matrix_values, all_sequences, mirror_diagonal=False)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Run demo if no arguments provided
        print("No arguments provided. Running demo with sample data...")
        demo_with_sample_data()
    else:
        main()