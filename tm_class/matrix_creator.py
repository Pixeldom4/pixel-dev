import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import os


def create_reference_matrix(input_file, reference_sequence=None, output_file=None, descending=False, plot=False, output_dir=None):
    """
    Create a matrix with reference sequence comparisons.
    
    Args:
        input_file: Path to input TSV file
        reference_sequence: Reference sequence accession (if None, uses most frequent)
        output_file: Output TSV file path (if None, uses input filename + '_matrix.tsv')
    """
    
    # Read the input TSV file
    df = pd.read_csv(input_file, sep='\t')
    
    # Check required columns
    required_cols = ['label1', 'label2', 'tm1_true', 'tm2_true', 'tm1_pred', 'tm2_pred']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # If no reference sequence specified, find the most frequent one
    if reference_sequence is None:
        # Count occurrences in both label1 and label2
        label1_counts = df['label1'].value_counts()
        label2_counts = df['label2'].value_counts()
        all_counts = label1_counts.add(label2_counts, fill_value=0)
        reference_sequence = all_counts.idxmax()
        print(f"Using most frequent sequence as reference: {reference_sequence}")
    
    # Filter data where reference sequence is involved
    ref_data = []
    
    # Case 1: Reference is in label1
    mask1 = df['label1'] == reference_sequence
    for _, row in df[mask1].iterrows():
        ref_data.append({
            'accession': row['label2'],
            'true_tm': row['tm2_true'],
            'pred_tm': row['tm2_pred']
        })
    
    # Case 2: Reference is in label2
    mask2 = df['label2'] == reference_sequence
    for _, row in df[mask2].iterrows():
        ref_data.append({
            'accession': row['label1'],
            'true_tm': row['tm1_true'],
            'pred_tm': row['tm1_pred']
        })
    
    # Create matrix DataFrame
    if not ref_data:
        raise ValueError(f"Reference sequence '{reference_sequence}' not found in data")
    
    matrix_df = pd.DataFrame(ref_data)
    
    # Remove duplicates (in case reference appears in both positions for same comparison)
    matrix_df = matrix_df.drop_duplicates(subset=['accession'])
    
    # Add over/under estimation column
    matrix_df['estimation'] = matrix_df.apply(lambda row: 
        'over' if row['pred_tm'] > row['true_tm'] else 'under', axis=1)
    
    # Sort by true_tm in ascending order (rank order)
    matrix_df = matrix_df.sort_values('true_tm', ascending=not descending).reset_index(drop=True)
    
    # Set output file path
    if output_file is None:
        base_name = input_file.rsplit('.', 1)[0]
        output_file = f"{base_name}_matrix_{reference_sequence.replace('/', '_').replace('|', '_')}.tsv"
    
    # Save matrix
    matrix_df.to_csv(output_file, sep='\t', index=False)
    
    print(f"Matrix created with {len(matrix_df)} sequences")
    print(f"Reference sequence: {reference_sequence}")
    print(f"Output saved to: {output_file}")
    
    # Display summary statistics
    print(f"\nSummary statistics:")
    print(f"True Tm range: {matrix_df['true_tm'].min():.2f} - {matrix_df['true_tm'].max():.2f}")
    print(f"Predicted Tm range: {matrix_df['pred_tm'].min():.2f} - {matrix_df['pred_tm'].max():.2f}")
    print(f"Mean absolute error: {abs(matrix_df['true_tm'] - matrix_df['pred_tm']).mean():.2f}")
    
    # Plot if requested
    if plot:
        plot_path = plot_predictions(matrix_df, reference_sequence, output_dir)
        print(f"Plot saved to: {plot_path}")
    
    return matrix_df


def plot_predictions(matrix_df, reference_sequence, output_dir=None):
    """
    Plot ranked sequences by true Tm with color coding for over/under estimation.
    Save the plot to file.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create x-axis positions (sequence rank order)
    x_positions = range(len(matrix_df))
    
    # Separate data by estimation type for coloring
    colors = ['red' if est == 'over' else 'blue' for est in matrix_df['estimation']]
    
    # Create scatter plot
    scatter = ax.scatter(x_positions, matrix_df['true_tm'], 
                        c=colors, alpha=0.7, s=50)
    
    # Create custom legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label=f'Overestimation (n={sum(matrix_df["estimation"] == "over")})')
    blue_patch = mpatches.Patch(color='blue', label=f'Underestimation (n={sum(matrix_df["estimation"] == "under")})')
    ax.legend(handles=[red_patch, blue_patch])
    
    # Formatting
    ax.set_xlabel('Sequence Rank (ordered by True Tm)', fontsize=12)
    ax.set_ylabel('True Tm (°C)', fontsize=12)
    ax.set_title(f'Ranked Sequences by True Tm\nReference: {reference_sequence}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show every 10th sequence label if there are many sequences
    if len(matrix_df) > 20:
        step = max(1, len(matrix_df) // 10)
        tick_positions = range(0, len(matrix_df), step)
        tick_labels = [matrix_df.iloc[i]['accession'][:10] + '...' if len(matrix_df.iloc[i]['accession']) > 10 
                      else matrix_df.iloc[i]['accession'] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    else:
        # Show all labels for smaller datasets
        ax.set_xticks(x_positions)
        ax.set_xticklabels([acc[:8] + '...' if len(acc) > 8 else acc for acc in matrix_df['accession']], 
                          rotation=45, ha='right')
    
    # Add statistics text
    mae = abs(matrix_df['true_tm'] - matrix_df['pred_tm']).mean()
    rmse = ((matrix_df['true_tm'] - matrix_df['pred_tm']) ** 2).mean() ** 0.5
    r_squared = matrix_df['true_tm'].corr(matrix_df['pred_tm']) ** 2
    
    stats_text = f'MAE: {mae:.2f}°C\nRMSE: {rmse:.2f}°C\nR²: {r_squared:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    ref_name = reference_sequence.replace('/', '_').replace('|', '_').replace('.', '_')
    plot_filename = f"prediction_plot_{ref_name}.png"
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        plot_path = os.path.join(output_dir, plot_filename)
    else:
        plot_path = plot_filename
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    return plot_path
    
    return matrix_df


def list_available_sequences(input_file, top_n=10):
    """
    List available sequences and their frequencies for reference selection.
    """
    df = pd.read_csv(input_file, sep='\t')
    
    # Count occurrences in both label columns
    label1_counts = df['label1'].value_counts()
    label2_counts = df['label2'].value_counts()
    all_counts = label1_counts.add(label2_counts, fill_value=0).sort_values(ascending=False)
    
    print(f"Top {top_n} most frequent sequences:")
    print("-" * 50)
    for i, (seq, count) in enumerate(all_counts.head(top_n).items(), 1):
        print(f"{i:2d}. {seq}: {int(count)} comparisons")


def main():
    parser = argparse.ArgumentParser(
        description='Create a matrix from pairwise sequence comparison data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available sequences
  python script.py input.tsv --list
  
  # Create matrix with most frequent sequence as reference
  python script.py input.tsv
  
  # Create matrix and plot, save both in a new folder
  python script.py input.tsv --reference WP_093954453.1 --plot --output-dir results_WP_093954453
  
  # Specify output filename and directory
  python script.py input.tsv --reference WP_093954453.1 --output my_matrix.tsv --output-dir analysis_results
  
  # Specify output file
  python script.py input.tsv --reference WP_093954453.1 --output my_matrix.tsv
        """
    )
    
    parser.add_argument('input_file', help='Input TSV file with pairwise comparison data')
    parser.add_argument('--reference', '-r', help='Reference sequence accession')
    parser.add_argument('--output', '-o', help='Output TSV file path')
    parser.add_argument('--descending', '-d', action='store_true',
                        help='Sort by true_tm in descending order (highest to lowest)')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Generate and save scatter plot of true vs predicted Tm values')
    parser.add_argument('--output-dir', '--output_dir', dest='output_dir', help='Output directory for all files')
    parser.add_argument('--list', '-l', action='store_true', 
                        help='List available sequences and their frequencies')
    
    args = parser.parse_args()
    
    try:
        if args.list:
            list_available_sequences(args.input_file)
        else:
            create_reference_matrix(args.input_file, args.reference, args.output, 
                                   args.descending, args.plot, args.output_dir)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()