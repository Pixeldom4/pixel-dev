#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
import os

def calculate_sequence_identity(seq1, seq2):
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1, seq2 = seq1[:min_len], seq2[:min_len]
    
    seq1, seq2 = seq1.upper(), seq2.upper()
    identical = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-' and b != '-')
    valid_positions = sum(1 for a, b in zip(seq1, seq2) if a != '-' and b != '-')
    
    return identical / valid_positions if valid_positions > 0 else 0.0

def read_prediction_data(filename):
    try:
        df = pd.read_csv(filename, sep='\t', header=None, skiprows=1)
        df.columns = ['ID', 'Tm', 'Predicted_Tm', 'Sequence']
        return df
    except Exception:
        return None

def generate_pairwise_data(df, output_file, max_pairs=None, min_pctid=None, max_pctid=None):
    proteins_data = []
    for _, row in df.iterrows():
        try:
            protein = {
                'id': str(row['ID']).strip(),
                'true_tm': float(row['Tm']),
                'pred_tm': float(row['Predicted_Tm']),
                'sequence': str(row['Sequence']).strip()
            }
            proteins_data.append(protein)
        except Exception:
            continue
    
    all_pairs = list(combinations(proteins_data, 2))
    
    if max_pairs and len(all_pairs) > max_pairs:
        np.random.shuffle(all_pairs)
        all_pairs = all_pairs[:max_pairs]
    
    pairwise_data = []
    
    for protein1, protein2 in tqdm(all_pairs):
        try:
            pctid = calculate_sequence_identity(protein1['sequence'], protein2['sequence'])
        except Exception:
            continue
        
        if min_pctid is not None and pctid < min_pctid:
            continue
        if max_pctid is not None and pctid > max_pctid:
            continue
        
        true_tm_diff = protein2['true_tm'] - protein1['true_tm']
        pred_tm_diff = protein2['pred_tm'] - protein1['pred_tm']
        
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
        return None
    
    result_df = pd.DataFrame(pairwise_data)
    result_df.to_csv(output_file, sep='\t', index=False)
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Generate pairwise melting temperature differences from a single TSV file")
    
    parser.add_argument("--input_tsv", 
                        type=str, required=True,
                        help="TSV file with prediction results (columns: ID, Tm, Predicted_Tm, Sequence)")
    
    parser.add_argument("--output", "-o",
                        type=str,
                        default="./results/",
                        help="Output file path")
    
    parser.add_argument("--seed", 
                        type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    if args.output.endswith('/'):
        input_basename = os.path.splitext(os.path.basename(args.input_tsv))[0]
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, f"{input_basename}_pairwise.tsv")
    else:
        output_file = args.output
    
    df = read_prediction_data(args.input_tsv)
    if df is None:
        return 1
    
    result_df = generate_pairwise_data(df=df, output_file=output_file)
    
    if result_df is not None:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())