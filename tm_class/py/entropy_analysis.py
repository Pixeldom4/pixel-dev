import pandas as pd
import numpy as np
import argparse
import os

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception:
        return None

def find_uncertainty_range_entropy(true_tm, estimation, bin_width=2.0, entropy_threshold=0.8):
    min_tm, max_tm = min(true_tm), max(true_tm)
    bins = np.arange(min_tm, max_tm + bin_width, bin_width)
    
    bin_entropies = []
    bin_centers = []
    bin_counts = []
    bin_over_props = []
    bin_starts = []
    bin_ends = []
    
    for i in range(len(bins) - 1):
        in_bin = [(tm >= bins[i] and tm < bins[i+1]) for tm in true_tm]
        bin_estimations = [estimation[j] for j in range(len(estimation)) if in_bin[j]]
        
        if len(bin_estimations) == 0:
            continue
            
        over_count = sum(1 for est in bin_estimations if est == 'over')
        under_count = len(bin_estimations) - over_count
        over_prop = over_count / len(bin_estimations)
        under_prop = 1 - over_prop
        
        if over_prop == 0 or under_prop == 0:
            entropy = 0
        else:
            entropy = -over_prop * np.log2(over_prop) - under_prop * np.log2(under_prop)
        
        bin_entropies.append(entropy)
        bin_centers.append((bins[i] + bins[i+1]) / 2)
        bin_starts.append(bins[i])
        bin_ends.append(bins[i+1])
        bin_counts.append(len(bin_estimations))
        bin_over_props.append(over_prop)
    
    uncertain_indices = [i for i, entropy in enumerate(bin_entropies) if entropy >= entropy_threshold]
    
    longest_consecutive_range = None
    if uncertain_indices:
        consecutive_sequences = []
        current_sequence = [uncertain_indices[0]]
        
        for i in range(1, len(uncertain_indices)):
            if uncertain_indices[i] == uncertain_indices[i-1] + 1:
                current_sequence.append(uncertain_indices[i])
            else:
                consecutive_sequences.append(current_sequence)
                current_sequence = [uncertain_indices[i]]
        consecutive_sequences.append(current_sequence)
        
        longest_sequence = max(consecutive_sequences, key=len)
        
        start_idx = longest_sequence[0]
        end_idx = longest_sequence[-1]
        longest_consecutive_range = (bin_starts[start_idx], bin_ends[end_idx])
    
    return {
        'uncertainty_range': longest_consecutive_range,
        'bin_centers': bin_centers,
        'bin_entropies': bin_entropies,
        'bin_counts': bin_counts,
        'bin_over_props': bin_over_props,
        'bin_starts': bin_starts,
        'bin_ends': bin_ends,
        'bins': bins
    }

def save_results_tsv(entropy_results, bin_width, entropy_threshold, output_path, reference_accession):
    uncertainty_start, uncertainty_end = entropy_results['uncertainty_range']
    bin_entropies = entropy_results['bin_entropies']
    bin_starts = entropy_results['bin_starts']
    bin_ends = entropy_results['bin_ends']
    
    if uncertainty_start is not None and uncertainty_end is not None:
        range_width = uncertainty_end - uncertainty_start
        
        uncertain_bins_in_range = sum(1 for i, entropy in enumerate(bin_entropies) 
                                    if entropy >= entropy_threshold and 
                                    bin_starts[i] >= uncertainty_start and 
                                    bin_ends[i] <= uncertainty_end)
        
        summary_data = [{
            'Reference_Accession': reference_accession,
            'Range_Start_Celsius': uncertainty_start,
            'Range_End_Celsius': uncertainty_end,
            'Range_Width_Celsius': range_width,
            'Bin_Width_Used': bin_width,
            'Entropy_Threshold': entropy_threshold,
            'Uncertain_Bins_Count': uncertain_bins_in_range,
            'Max_Entropy_Achieved': max(bin_entropies) if bin_entropies else 0
        }]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, sep='\t', index=False, float_format='%.3f')

def main():
    parser = argparse.ArgumentParser(description='Detect uncertainty ranges using entropy analysis')
    parser.add_argument('input_file', help='Input TSV file path')
    parser.add_argument('--bin-width', '-b', type=float, default=2.0, help='Bin width for entropy calculation (default: 2.0°C)')
    parser.add_argument('--entropy-threshold', '-t', type=float, default=0.8, help='Entropy threshold for uncertainty (default: 0.8)')
    parser.add_argument('--output', '-o', 
                        type=str,
                        default='./results/', 
                        help='Output file path')
    
    args = parser.parse_args()
    
    filename = os.path.basename(args.input_file)
    if '_matrix_' in filename:
        reference_accession = filename.split('_matrix_')[1].replace('.tsv', '')
    else:
        reference_accession = 'Unknown'
    
    if args.output.endswith('/'):
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, 'uncertainty_range_summary.tsv')
    else:
        output_file = args.output
    
    df = load_data(args.input_file)
    
    if df is None:
        return
    
    required_columns = ['accession', 'true_tm', 'pred_tm', 'estimation']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return
    
    unique_estimations = df['estimation'].unique()
    
    if not all(est in ['over', 'under'] for est in unique_estimations):
        return
    
    df_sorted = df.sort_values('true_tm').reset_index(drop=True)
    
    entropy_results = find_uncertainty_range_entropy(
        df_sorted['true_tm'].tolist(), 
        df_sorted['estimation'].tolist(), 
        args.bin_width, 
        args.entropy_threshold
    )
    
    save_results_tsv(entropy_results, args.bin_width, args.entropy_threshold, output_file, reference_accession)

if __name__ == "__main__":
    main()