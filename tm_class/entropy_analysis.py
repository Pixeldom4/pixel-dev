import pandas as pd
import numpy as np
import argparse
import os

def load_data(file_path):
    """Load TSV data from file."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def find_uncertainty_range_entropy(true_tm, estimation, bin_width=2.0, entropy_threshold=0.8):
    """Use entropy to measure prediction uncertainty across Tm bins."""
    
    # Create bins
    min_tm, max_tm = min(true_tm), max(true_tm)
    bins = np.arange(min_tm, max_tm + bin_width, bin_width)
    
    bin_entropies = []
    bin_centers = []
    bin_counts = []
    bin_over_props = []
    bin_starts = []
    bin_ends = []
    
    for i in range(len(bins) - 1):
        # Find points in this bin
        in_bin = [(tm >= bins[i] and tm < bins[i+1]) for tm in true_tm]
        bin_estimations = [estimation[j] for j in range(len(estimation)) if in_bin[j]]
        
        if len(bin_estimations) == 0:
            continue
            
        # Calculate proportions
        over_count = sum(1 for est in bin_estimations if est == 'over')
        under_count = len(bin_estimations) - over_count
        over_prop = over_count / len(bin_estimations)
        under_prop = 1 - over_prop
        
        # Calculate entropy
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
    
    # Find longest consecutive sequence of uncertain bins
    uncertain_indices = [i for i, entropy in enumerate(bin_entropies) if entropy >= entropy_threshold]
    
    longest_consecutive_range = None
    if uncertain_indices:
        # Find consecutive sequences
        consecutive_sequences = []
        current_sequence = [uncertain_indices[0]]
        
        for i in range(1, len(uncertain_indices)):
            if uncertain_indices[i] == uncertain_indices[i-1] + 1:
                current_sequence.append(uncertain_indices[i])
            else:
                consecutive_sequences.append(current_sequence)
                current_sequence = [uncertain_indices[i]]
        consecutive_sequences.append(current_sequence)
        
        # Find the longest sequence
        longest_sequence = max(consecutive_sequences, key=len)
        
        # Get the range for the longest sequence
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

def print_entropy_summary(entropy_results, bin_width, entropy_threshold):
    """Print summary of entropy analysis."""
    
    uncertainty_start, uncertainty_end = entropy_results['uncertainty_range']
    bin_centers = entropy_results['bin_centers']
    bin_entropies = entropy_results['bin_entropies']
    bin_counts = entropy_results['bin_counts']
    bin_over_props = entropy_results['bin_over_props']
    bin_starts = entropy_results['bin_starts']
    bin_ends = entropy_results['bin_ends']
    
    print("\n" + "="*60)
    print("ENTROPY-BASED UNCERTAINTY ANALYSIS SUMMARY")
    print("="*60)
    
    if uncertainty_start is not None and uncertainty_end is not None:
        print(f"🎯 LONGEST CONSECUTIVE UNCERTAINTY RANGE: {uncertainty_start:.1f}°C - {uncertainty_end:.1f}°C")
        range_width = uncertainty_end - uncertainty_start
        print(f"   Range width: {range_width:.1f}°C")
    else:
        print("❌ No consecutive uncertainty range detected above threshold")
    
    print(f"\nParameters used:")
    print(f"  - Bin width: {bin_width}°C")
    print(f"  - Entropy threshold: {entropy_threshold}")
    
    print(f"\nBin-by-bin breakdown:")
    print(f"{'Tm_Range':<15} {'Count':<8} {'Over_Percent':<12} {'Entropy':<10} {'Status':<12}")
    print("-" * 65)
    
    for i, center in enumerate(bin_centers):
        tm_start = bin_starts[i]
        tm_end = bin_ends[i]
        count = bin_counts[i]
        over_pct = bin_over_props[i] * 100
        entropy = bin_entropies[i]
        status = "UNCERTAIN" if entropy >= entropy_threshold else "Confident"
        
        print(f"{tm_start:.1f}-{tm_end:.1f}°C    {count:<8} {over_pct:<11.1f}% {entropy:<9.3f} {status}")
    
    # Find the most uncertain bin
    if bin_entropies:
        max_entropy_idx = np.argmax(bin_entropies)
        max_entropy = bin_entropies[max_entropy_idx]
        max_entropy_start = bin_starts[max_entropy_idx]
        max_entropy_end = bin_ends[max_entropy_idx]
        
        print(f"\n🔥 Most uncertain bin: {max_entropy_start:.1f}-{max_entropy_end:.1f}°C")
        print(f"   Entropy: {max_entropy:.3f}")
        print(f"   Overestimate proportion: {bin_over_props[max_entropy_idx]:.3f}")

def save_results_tsv(entropy_results, bin_width, entropy_threshold, output_dir):
    """Save results as TSV files."""
    
    uncertainty_start, uncertainty_end = entropy_results['uncertainty_range']
    bin_centers = entropy_results['bin_centers']
    bin_entropies = entropy_results['bin_entropies']
    bin_counts = entropy_results['bin_counts']
    bin_over_props = entropy_results['bin_over_props']
    bin_starts = entropy_results['bin_starts']
    bin_ends = entropy_results['bin_ends']
    
    # Create bin analysis TSV
    bin_data = []
    for i in range(len(bin_centers)):
        tm_start = bin_starts[i]
        tm_end = bin_ends[i]
        count = bin_counts[i]
        over_pct = bin_over_props[i] * 100
        entropy = bin_entropies[i]
        status = "UNCERTAIN" if entropy >= entropy_threshold else "Confident"
        
        bin_data.append({
            'Tm_Range': f"{tm_start:.1f}-{tm_end:.1f}°C",
            'Tm_Start': tm_start,
            'Tm_End': tm_end,
            'Count': count,
            'Over_Percent': over_pct,
            'Entropy': entropy,
            'Status': status
        })
    
    bin_df = pd.DataFrame(bin_data)
    bin_tsv_path = os.path.join(output_dir, 'bin_analysis.tsv')
    bin_df.to_csv(bin_tsv_path, sep='\t', index=False, float_format='%.3f')
    print(f"Bin analysis saved to: {bin_tsv_path}")
    
    # Create uncertainty range summary TSV
    if uncertainty_start is not None and uncertainty_end is not None:
        range_width = uncertainty_end - uncertainty_start
        
        # Count uncertain bins in the consecutive range
        uncertain_bins_in_range = sum(1 for i, entropy in enumerate(bin_entropies) 
                                    if entropy >= entropy_threshold and 
                                    bin_starts[i] >= uncertainty_start and 
                                    bin_ends[i] <= uncertainty_end)
        
        summary_data = [{
            'Analysis_Type': 'Longest_Consecutive_Uncertainty_Range',
            'Range_Start_Celsius': uncertainty_start,
            'Range_End_Celsius': uncertainty_end,
            'Range_Width_Celsius': range_width,
            'Bin_Width_Used': bin_width,
            'Entropy_Threshold': entropy_threshold,
            'Uncertain_Bins_Count': uncertain_bins_in_range,
            'Max_Entropy_Achieved': max(bin_entropies) if bin_entropies else 0
        }]
        
        summary_df = pd.DataFrame(summary_data)
        summary_tsv_path = os.path.join(output_dir, 'uncertainty_range_summary.tsv')
        summary_df.to_csv(summary_tsv_path, sep='\t', index=False, float_format='%.3f')
        print(f"Uncertainty range summary saved to: {summary_tsv_path}")
    else:
        print("No uncertainty range to save - no consecutive uncertain bins found.")

def main():
    parser = argparse.ArgumentParser(description='Detect uncertainty ranges using entropy analysis')
    parser.add_argument('input_file', help='Input TSV file path')
    parser.add_argument('--bin-width', '-b', type=float, default=2.0, help='Bin width for entropy calculation (default: 2.0°C)')
    parser.add_argument('--entropy-threshold', '-t', type=float, default=0.8, help='Entropy threshold for uncertainty (default: 0.8)')
    parser.add_argument('--output-dir', '-o', help='Directory to save results (optional)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    df = load_data(args.input_file)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Validate required columns
    required_columns = ['accession', 'true_tm', 'pred_tm', 'estimation']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"Data loaded successfully! Shape: {df.shape}")
    
    # Check estimation values
    unique_estimations = df['estimation'].unique()
    print(f"Estimation categories found: {unique_estimations}")
    
    if not all(est in ['over', 'under'] for est in unique_estimations):
        print("Warning: Expected 'over' and 'under' in estimation column")
    
    # Ensure data is sorted by true_tm
    df_sorted = df.sort_values('true_tm').reset_index(drop=True)
    
    # Calculate entropy analysis
    entropy_results = find_uncertainty_range_entropy(
        df_sorted['true_tm'].tolist(), 
        df_sorted['estimation'].tolist(), 
        args.bin_width, 
        args.entropy_threshold
    )
    
    # Print summary
    print_entropy_summary(entropy_results, args.bin_width, args.entropy_threshold)
    
    # Save TSV results if output directory provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        save_results_tsv(entropy_results, args.bin_width, args.entropy_threshold, args.output_dir)

if __name__ == "__main__":
    main()