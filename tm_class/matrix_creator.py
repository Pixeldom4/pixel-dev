import pandas as pd
import argparse
import sys


def create_reference_matrix(input_file, reference_sequence=None, output_file=None, descending=False):
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
  
  # Create matrix with specific reference sequence (descending order)
  python script.py input.tsv --reference WP_093954453.1 --descending
  
  # Specify output file
  python script.py input.tsv --reference WP_093954453.1 --output my_matrix.tsv
        """
    )
    
    parser.add_argument('input_file', help='Input TSV file with pairwise comparison data')
    parser.add_argument('--reference', '-r', help='Reference sequence accession')
    parser.add_argument('--output', '-o', help='Output TSV file path')
    parser.add_argument('--descending', '-d', action='store_true',
                        help='Sort by true_tm in descending order (highest to lowest)')
    parser.add_argument('--list', '-l', action='store_true', 
                        help='List available sequences and their frequencies')
    
    args = parser.parse_args()
    
    try:
        if args.list:
            list_available_sequences(args.input_file)
        else:
            create_reference_matrix(args.input_file, args.reference, args.output, args.descending)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()