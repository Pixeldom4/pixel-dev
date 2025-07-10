#!/usr/bin/env python3
# The usecase of this script is to convert pairwise melting temperature difference predictions, specifically from pairwise differences outputed by petai into degree C

import argparse
import pandas as pd
import re

def clean_tensor_values(value):
    """Clean tensor() values from the data"""
    if isinstance(value, str) and 'tensor(' in value:
        # Extract number from tensor(X) format
        match = re.search(r'tensor\((\d+)\)', value)
        if match:
            return int(match.group(1))
    return value

def convert_to_accession_predictions(input_file, output_file):
    """
    Convert pairwise predictions to individual protein melting temperatures.
    Output format: accession, tm_pred
    """
    
    print(f"Reading predictions from {input_file}...")
    df = pd.read_csv(input_file, sep='\t')
    
    print(f"Found {len(df)} pairwise predictions")
    
    # Clean any tensor values in label columns
    df['label1'] = df['label1'].apply(clean_tensor_values)
    df['label2'] = df['label2'].apply(clean_tensor_values)
    
    # Calculate predicted tm2 for each pair
    df['predicted_tm2'] = df['tm1'] + df['y_pred']
    
    # Create dictionary to store predictions for each protein
    protein_predictions = {}
    
    # For each pair, we get one prediction for protein2
    for _, row in df.iterrows():
        accession1 = str(row['label1'])
        accession2 = str(row['label2'])
        tm1 = row['tm1']
        predicted_tm2 = row['predicted_tm2']
        
        # Store the known tm1 for protein1
        if accession1 not in protein_predictions:
            protein_predictions[accession1] = []
        protein_predictions[accession1].append(tm1)
        
        # Store the predicted tm2 for protein2
        if accession2 not in protein_predictions:
            protein_predictions[accession2] = []
        protein_predictions[accession2].append(predicted_tm2)
    
    # Calculate average prediction for each protein
    results = []
    for accession, temps in protein_predictions.items():
        avg_temp = sum(temps) / len(temps)
        results.append({
            'accession': accession,
            'tm_pred': round(avg_temp, 4)
        })
    
    # Create output dataframe
    output_df = pd.DataFrame(results)
    output_df = output_df.sort_values('accession')
    
    print(f"Generated predictions for {len(output_df)} unique proteins")
    
    # Save output
    print(f"Writing predictions to {output_file}...")
    output_df.to_csv(output_file, sep='\t', index=False)
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"Predicted TM range: {output_df['tm_pred'].min():.2f} - {output_df['tm_pred'].max():.2f} °C")
    print(f"Mean predicted TM: {output_df['tm_pred'].mean():.2f} °C")
    print(f"Number of proteins: {len(output_df)}")
    
    return output_df

def main():
    parser = argparse.ArgumentParser(
        description="Convert pairwise TM predictions to individual protein predictions",
        epilog="Example: python convert_tm_predictions.py --input predictions.tsv --output protein_temps.tsv"
    )
    
    parser.add_argument("--input", "-i", 
                        type=str, 
                        required=True,
                        help="Input TSV file with pairwise difference predictions")
    
    parser.add_argument("--output", "-o", 
                        type=str, 
                        required=True,
                        help="Output TSV file with format: accession, tm_pred")
    
    args = parser.parse_args()
    
    # Check input file exists
    try:
        with open(args.input, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    # Convert predictions
    try:
        convert_to_accession_predictions(args.input, args.output)
        print("✓ Conversion completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1

if __name__ == "__main__":
    exit(main())