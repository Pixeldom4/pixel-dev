import pandas as pd
import argparse
import sys
import os


def create_reference_matrix(input_file, reference_sequence, output_file):
    df = pd.read_csv(input_file, sep='\t', low_memory=False)
    
    required_cols = ['label1', 'label2', 'tm1_true', 'tm2_true', 'tm1_pred', 'tm2_pred']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    ref_data = []
    
    mask1 = df['label1'] == reference_sequence
    for _, row in df[mask1].iterrows():
        ref_data.append({
            'accession': row['label2'],
            'true_tm': row['tm2_true'],
            'pred_tm': row['tm2_pred']
        })
    
    mask2 = df['label2'] == reference_sequence
    for _, row in df[mask2].iterrows():
        ref_data.append({
            'accession': row['label1'],
            'true_tm': row['tm1_true'],
            'pred_tm': row['tm1_pred']
        })
    
    if not ref_data:
        raise ValueError(f"Reference sequence '{reference_sequence}' not found in data")
    
    matrix_df = pd.DataFrame(ref_data)
    matrix_df = matrix_df.drop_duplicates(subset=['accession'])
    
    matrix_df['estimation'] = matrix_df.apply(lambda row: 
        'over' if row['pred_tm'] > row['true_tm'] else 'under', axis=1)
    
    matrix_df = matrix_df.sort_values('true_tm', ascending=True).reset_index(drop=True)
    
    matrix_df.to_csv(output_file, sep='\t', index=False)
    
    return matrix_df


def main():
    parser = argparse.ArgumentParser(description='Create a matrix from pairwise sequence comparison data')
    
    parser.add_argument('input_file', help='Input TSV file with pairwise comparison data')
    parser.add_argument('--reference', '-r', required=True, help='Reference sequence accession')
    parser.add_argument('--output', '-o', 
                        type=str,
                        default='./results/',
                        help='Output file path')
    
    args = parser.parse_args()
    
    if args.output.endswith('/'):
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, f"{base_name}_matrix_{args.reference.replace('/', '_').replace('|', '_')}.tsv")
    else:
        output_file = args.output
    
    try:
        create_reference_matrix(args.input_file, args.reference, output_file)
    except Exception as e:
        sys.exit(1)


if __name__ == '__main__':
    main()