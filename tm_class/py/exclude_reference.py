#!/usr/bin/env python3

import argparse
import pandas as pd
import os

def exclude_reference(input_file, reference_sequence, output_file):
    df = pd.read_csv(input_file, sep='\t', header=None, skiprows=1)
    df.columns = ['ID', 'Tm', 'Predicted_Tm', 'Sequence']
    
    original_count = len(df)
    df_filtered = df[df['ID'] != reference_sequence]
    filtered_count = len(df_filtered)
    
    if original_count == filtered_count:
        print(f"Warning: Reference sequence '{reference_sequence}' not found in dataset")
    
    df_filtered.to_csv(output_file, sep='\t', index=False)
    return df_filtered

def main():
    parser = argparse.ArgumentParser(description='Remove reference sequence from dataset to prevent leakage')
    
    parser.add_argument('--input_tsv', required=True, help='Input TSV file with predictions')
    parser.add_argument('--reference', '-r', required=True, help='Reference sequence ID to exclude')
    parser.add_argument('--output', '-o', 
                        type=str,
                        default='./results/',
                        help='Output file path')
    
    args = parser.parse_args()
    
    if args.output.endswith('/'):
        input_basename = os.path.splitext(os.path.basename(args.input_tsv))[0]
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, f"{input_basename}_no_ref.tsv")
    else:
        output_file = args.output
    
    exclude_reference(args.input_tsv, args.reference, output_file)

if __name__ == "__main__":
    main()