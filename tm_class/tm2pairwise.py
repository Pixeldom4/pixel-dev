#!/usr/bin/env python3
"""
Generate pairwise melting temperature differences from a single TSV file.
Creates pairwise comparisons with sequence identity calculations.
"""

import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
import os

def calculate_sequence_identity(seq1, seq2):
    """Calculate percentage sequence identity between two sequences"""
    if len(seq1) != len(seq2):
        # For unaligned sequences, use shorter length
        min_len = min(len(seq1), len(seq2))
        seq1, seq2 = seq1[:min_len], seq2[:min_len]
    
    seq1, seq2 = seq1.upper(), seq2.upper()
    identical = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-' and b != '-')
    valid_positions = sum(1 for a, b in zip(seq1, seq2) if a != '-' and b != '-')
    
    return identical / valid_positions if valid_positions > 0 else 0.0

def read_prediction_data(filename):
    """Read prediction data from TSV file"""
    try:
        # Try different separators
        try:
            df = pd.read_csv(filename, sep='\t')
        except:
            df = pd.read_csv(filename)
        
        print(f"Loaded {len(df)} proteins from {filename}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def generate_pairwise_data(df, id_col, true_tm_col, pred_tm_col, sequence_col, output_file,
                          max_pairs=None, min_pctid=None, max_pctid=None):
    """
    Generate pairwise data from prediction results
    
    Args:
        df: DataFrame with protein data
        id_col: Column name for protein IDs
        true_tm_col: Column name for true melting temperatures
        pred_tm_col: Column name for predicted melting temperatures
        sequence_col: Column name for sequences
        output_file: Output file path
        max_pairs: Maximum number of pairs to generate
        min_pctid: Minimum sequence identity threshold
        max_pctid: Maximum sequence identity threshold
    """
    
    # Convert DataFrame to list of dictionaries for easier processing
    proteins_data = []
    for _, row in df.iterrows():
        try:
            protein = {
                'id': str(row[id_col]).strip(),
                'true_tm': float(row[true_tm_col]),
                'pred_tm': float(row[pred_tm_col]),
                'sequence': str(row[sequence_col]).strip()
            }
            proteins_data.append(protein)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue
    
    # Generate all possible pairs
    all_pairs = list(combinations(proteins_data, 2))
    print(f"Total possible pairs: {len(all_pairs)}")
    
    # Limit pairs if requested
    if max_pairs and len(all_pairs) > max_pairs:
        print(f"Limiting to {max_pairs} random pairs")
        np.random.shuffle(all_pairs)
        all_pairs = all_pairs[:max_pairs]
    
    # Generate pairwise data
    pairwise_data = []
    
    print("Generating pairwise comparisons...")
    for protein1, protein2 in tqdm(all_pairs):
        # Calculate sequence identity
        try:
            pctid = calculate_sequence_identity(protein1['sequence'], protein2['sequence'])
        except Exception as e:
            print(f"Skipping pair {protein1['id']}-{protein2['id']}: {e}")
            continue
        
        # Apply sequence identity filters
        if min_pctid is not None and pctid < min_pctid:
            continue
        if max_pctid is not None and pctid > max_pctid:
            continue
        
        # Calculate temperature differences
        true_tm_diff = protein2['true_tm'] - protein1['true_tm']
        pred_tm_diff = protein2['pred_tm'] - protein1['pred_tm']
        
        # Create data row
        row = {
            'y_true': round(true_tm_diff, 6),
            'y_pred': round(pred_tm_diff, 6),
            'pctid': round(pctid, 4),
            'label1': protein1['id'],
            'label2': protein2['id'],
            'tm1_true': protein1['true_tm'],
            'tm2_true': protein2['true_tm'],
            'tm1_pred': protein1['pred_tm'],
            'tm2_pred': protein2['pred_tm']
        }
        
        pairwise_data.append(row)
    
    if len(pairwise_data) == 0:
        print("No pairs met the filtering criteria")
        return None
    
    # Create DataFrame
    result_df = pd.DataFrame(pairwise_data)
    
    print(f"Generated {len(result_df)} pairwise comparisons")
    print(f"True temp difference range: {result_df['y_true'].min():.2f} to {result_df['y_true'].max():.2f}")
    print(f"Predicted temp difference range: {result_df['y_pred'].min():.2f} to {result_df['y_pred'].max():.2f}")
    print(f"Sequence identity range: {result_df['pctid'].min():.4f} to {result_df['pctid'].max():.4f}")
    
    # Save to file
    result_df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved pairwise data to {output_file}")
    
    # Print summary statistics
    mean_true_diff = result_df['y_true'].mean()
    std_true_diff = result_df['y_true'].std()
    mean_pred_diff = result_df['y_pred'].mean()
    std_pred_diff = result_df['y_pred'].std()
    
    print(f"\nSummary Statistics:")
    print(f"Mean true temperature difference: {mean_true_diff:.2f} °C")
    print(f"Std true temperature difference: {std_true_diff:.2f} °C")
    print(f"Mean predicted temperature difference: {mean_pred_diff:.2f} °C")
    print(f"Std predicted temperature difference: {std_pred_diff:.2f} °C")
    print(f"Mean sequence identity: {result_df['pctid'].mean():.4f}")
    print(f"Std sequence identity: {result_df['pctid'].std():.4f}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise melting temperature differences from a single TSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python generate_pairwise_single.py --input predictions.tsv --output pairwise_data.tsv
    
    # With filtering
    python generate_pairwise_single.py --input predictions.tsv --output pairwise_data.tsv --min_pctid 0.1 --max_pctid 0.9 --max_pairs 5000
    
    # Custom column names
    python generate_pairwise_single.py --input predictions.tsv --output pairwise_data.tsv --id_col "Protein_ID" --true_tm_col "Experimental_Tm"
        """
    )
    
    parser.add_argument("--input", "-i", 
                        type=str, required=True,
                        help="TSV file with prediction results")
    
    parser.add_argument("--output", "-o", 
                        type=str, required=True,
                        help="Output TSV file for pairwise data")
    
    parser.add_argument("--id_col", 
                        type=str, default="ID",
                        help="Column name for protein IDs (default: 'ID')")
    
    parser.add_argument("--true_tm_col", 
                        type=str, default="Tm",
                        help="Column name for true melting temperatures (default: 'Tm')")
    
    parser.add_argument("--pred_tm_col", 
                        type=str, default="Predicted Melting Point",
                        help="Column name for predicted melting temperatures (default: 'Predicted Melting Point')")
    
    parser.add_argument("--sequence_col", 
                        type=str, default="Sequence",
                        help="Column name for protein sequences (default: 'Sequence')")
    
    parser.add_argument("--max_pairs", 
                        type=int, default=None,
                        help="Maximum number of pairs to generate (default: all)")
    
    parser.add_argument("--min_pctid", 
                        type=float, default=None,
                        help="Minimum sequence identity threshold (0.0-1.0)")
    
    parser.add_argument("--max_pctid", 
                        type=float, default=None,
                        help="Maximum sequence identity threshold (0.0-1.0)")
    
    parser.add_argument("--seed", 
                        type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Read input data
    print("Reading prediction data...")
    df = read_prediction_data(args.input)
    if df is None:
        return 1
    
    # Validate column names
    required_cols = [args.id_col, args.true_tm_col, args.pred_tm_col, args.sequence_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate pairwise data
    try:
        result_df = generate_pairwise_data(
            df=df,
            id_col=args.id_col,
            true_tm_col=args.true_tm_col,
            pred_tm_col=args.pred_tm_col,
            sequence_col=args.sequence_col,
            output_file=args.output,
            max_pairs=args.max_pairs,
            min_pctid=args.min_pctid,
            max_pctid=args.max_pctid
        )
        
        if result_df is not None:
            print(f"\nSuccessfully generated pairwise data with {len(result_df)} pairs")
            return 0
        else:
            print("Failed to generate pairwise data")
            return 1
            
    except Exception as e:
        print(f"Error generating pairwise data: {e}")
        return 1

if __name__ == "__main__":
    exit(main())