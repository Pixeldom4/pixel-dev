#!/usr/bin/env python3
"""
Generate separate pairwise melting temperature data files for multiple prediction tools.
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

def read_benchmark_data(filename):
    """Read benchmark data with multiple prediction tools"""
    try:
        df = pd.read_csv(filename, sep='\t')
        print(f"Loaded {len(df)} proteins from {filename}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def generate_pairwise_for_tool(df, tool_name, accession_col, true_tm_col, pred_tm_col, sequence_col,
                              output_dir, max_pairs=None, min_pctid=None, max_pctid=None):
    """
    Generate pairwise data for a specific prediction tool
    
    Args:
        df: DataFrame with protein data
        tool_name: Name of the prediction tool
        accession_col: Column name for accessions
        true_tm_col: Column name for true melting temperatures
        pred_tm_col: Column name for predicted melting temperatures
        sequence_col: Column name for sequences
        output_dir: Output directory
        max_pairs: Maximum number of pairs to generate
        min_pctid: Minimum sequence identity threshold
        max_pctid: Maximum sequence identity threshold
    """
    
    # Convert DataFrame to list of dictionaries for easier processing
    proteins_data = []
    for _, row in df.iterrows():
        protein = {
            'accession': str(row[accession_col]).strip(),  # Ensure it's a clean string
            'true_tm': float(row[true_tm_col]),
            'pred_tm': float(row[pred_tm_col]),
            'sequence': str(row[sequence_col]).strip()
        }
        proteins_data.append(protein)
    
    # Generate all possible pairs
    all_pairs = list(combinations(proteins_data, 2))
    print(f"\n=== Processing {tool_name} ===")
    print(f"Total possible pairs: {len(all_pairs)}")
    
    # Limit pairs if requested
    if max_pairs and len(all_pairs) > max_pairs:
        print(f"Limiting to {max_pairs} random pairs")
        np.random.shuffle(all_pairs)
        all_pairs = all_pairs[:max_pairs]
    
    # Generate pairwise data
    pairwise_data = []
    
    print(f"Generating pairwise comparisons for {tool_name}...")
    for protein1, protein2 in tqdm(all_pairs):
        # Calculate sequence identity
        try:
            pctid = calculate_sequence_identity(protein1['sequence'], protein2['sequence'])
        except Exception as e:
            print(f"Skipping pair {protein1['accession']}-{protein2['accession']}: {e}")
            continue
        
        # Apply sequence identity filters
        if min_pctid is not None and pctid < min_pctid:
            continue
        if max_pctid is not None and pctid > max_pctid:
            continue
        
        # Calculate temperature differences
        true_tm_diff = protein2['true_tm'] - protein1['true_tm']
        pred_tm_diff = protein2['pred_tm'] - protein1['pred_tm']
        
        # Create data row matching your format
        row = {
            'y_true': round(true_tm_diff, 6),
            'y_pred': round(pred_tm_diff, 6),
            'pctid': round(pctid, 4),
            'label1': protein1['accession'],  # This should now be a clean string
            'label2': protein2['accession'],  # This should now be a clean string
            'tm1': protein1['true_tm'],
            'tm2': protein2['true_tm']
        }
        
        pairwise_data.append(row)
    
    if len(pairwise_data) == 0:
        print(f"No pairs met the filtering criteria for {tool_name}")
        return None
    
    # Create DataFrame
    result_df = pd.DataFrame(pairwise_data)
    
    print(f"Generated {len(result_df)} pairwise comparisons for {tool_name}")
    print(f"True temp difference range: {result_df['y_true'].min():.2f} to {result_df['y_true'].max():.2f}")
    print(f"Pred temp difference range: {result_df['y_pred'].min():.2f} to {result_df['y_pred'].max():.2f}")
    print(f"Sequence identity range: {result_df['pctid'].min():.4f} to {result_df['pctid'].max():.4f}")
    
    # Save to file
    output_file = os.path.join(output_dir, f"pairwise_{tool_name.lower()}.tsv")
    result_df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved {tool_name} pairwise data to {output_file}")
    
    # Print summary statistics
    correlation = result_df['y_true'].corr(result_df['y_pred'])
    rmse = np.sqrt(((result_df['y_true'] - result_df['y_pred']) ** 2).mean())
    
    print(f"\n{tool_name} Performance Summary:")
    print(f"Correlation (r): {correlation:.4f}")
    print(f"RMSE: {rmse:.2f} °C")
    print(f"Mean absolute error: {abs(result_df['y_true'] - result_df['y_pred']).mean():.2f} °C")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(
        description="Generate separate pairwise data files for multiple melting temperature prediction tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python generate_pairwise_benchmark.py --input benchmark_results.tsv --output_dir pairwise_data/
    
    # With filtering
    python generate_pairwise_benchmark.py --input benchmark_results.tsv --output_dir pairwise_data/ --min_pctid 0.1 --max_pctid 0.9 --max_pairs 5000
    
    # Custom column names
    python generate_pairwise_benchmark.py --input benchmark_results.tsv --output_dir pairwise_data/ --accession_col "Accession code" --true_tm_col "TmB"
        """
    )
    
    parser.add_argument("--input", "-i", 
                        type=str, required=True,
                        help="TSV file with benchmark results (multiple prediction tools)")
    
    parser.add_argument("--output_dir", "-o", 
                        type=str, required=True,
                        help="Output directory for pairwise data files")
    
    parser.add_argument("--accession_col", 
                        type=str, default="Accession code",
                        help="Column name for protein accessions (default: 'Accession code')")
    
    parser.add_argument("--true_tm_col", 
                        type=str, default="TmB",
                        help="Column name for true melting temperatures (default: 'TmB')")
    
    parser.add_argument("--sequence_col", 
                        type=str, default="Sequence",
                        help="Column name for protein sequences (default: 'Sequence')")
    
    parser.add_argument("--prediction_tools", 
                        type=str, nargs='+', 
                        default=["petai", "TemBERTure", "DeepStabP"],
                        help="List of prediction tool column names (default: petai TemBERTure DeepStabP)")
    
    parser.add_argument("--max_pairs", 
                        type=int, default=None,
                        help="Maximum number of pairs per tool (default: all)")
    
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
    print("Reading benchmark data...")
    df = read_benchmark_data(args.input)
    if df is None:
        return 1
    
    # Validate column names
    required_cols = [args.accession_col, args.true_tm_col, args.sequence_col] + args.prediction_tools
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processed {len(df)} proteins")
    
    # Generate pairwise data for each prediction tool
    results = {}
    for tool in args.prediction_tools:
        try:
            result_df = generate_pairwise_for_tool(
                df=df,
                tool_name=tool,
                accession_col=args.accession_col,
                true_tm_col=args.true_tm_col,
                pred_tm_col=tool,
                sequence_col=args.sequence_col,
                output_dir=args.output_dir,
                max_pairs=args.max_pairs,
                min_pctid=args.min_pctid,
                max_pctid=args.max_pctid
            )
            if result_df is not None:
                results[tool] = result_df
        except Exception as e:
            print(f"Error processing {tool}: {e}")
    
    # Generate summary comparison
    if len(results) > 1:
        print(f"\n=== TOOL COMPARISON SUMMARY ===")
        for tool, df_result in results.items():
            correlation = df_result['y_true'].corr(df_result['y_pred'])
            rmse = np.sqrt(((df_result['y_true'] - df_result['y_pred']) ** 2).mean())
            print(f"{tool:12s}: r={correlation:6.3f}, RMSE={rmse:6.2f}°C, n={len(df_result):5d} pairs")
    
    return 0

if __name__ == "__main__":
    exit(main())