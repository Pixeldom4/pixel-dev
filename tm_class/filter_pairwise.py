#!/usr/bin/env python3
"""
Filter pairwise data by removing rows where y_true is within a certain range.
This removes small perturbations to focus on larger temperature differences.
"""

import argparse
import pandas as pd
import numpy as np

def filter_pairwise_data(input_file, output_file, threshold, mode='range'):
    """
    Filter pairwise data by y_true threshold
    
    Args:
        input_file: Input TSV file with pairwise data
        output_file: Output TSV file for filtered data
        threshold: Threshold value for filtering y_true
        mode: Filtering mode - 'range', 'above', 'below', or 'abs'
    """
    
    try:
        # Read the data
        df = pd.read_csv(input_file, sep='\t')
        print(f"Loaded {len(df)} rows from {input_file}")
        
        # Show initial statistics
        print(f"\nInitial y_true statistics:")
        print(f"  Range: {df['y_true'].min():.2f} to {df['y_true'].max():.2f}")
        print(f"  Mean: {df['y_true'].mean():.2f}")
        print(f"  Std: {df['y_true'].std():.2f}")
        print(f"  Median: {df['y_true'].median():.2f}")
        
        # Apply filtering based on mode
        if mode == 'range':
            # Remove values within range [-threshold, +threshold]
            filtered_df = df[(df['y_true'] < -abs(threshold)) | (df['y_true'] > abs(threshold))]
            print(f"\nFiltering: keeping rows where y_true < -{abs(threshold)} OR y_true > {abs(threshold)}")
            print(f"Removing small perturbations in range [{-abs(threshold)}, {abs(threshold)}]")
        
        elif mode == 'abs':
            # Keep values where |y_true| >= threshold
            filtered_df = df[np.abs(df['y_true']) >= abs(threshold)]
            print(f"\nFiltering: keeping rows where |y_true| >= {abs(threshold)}")
        
        elif mode == 'above':
            # Remove values above threshold (original behavior)
            filtered_df = df[df['y_true'] <= threshold]
            print(f"\nFiltering: keeping rows where y_true <= {threshold}")
        
        elif mode == 'below':
            # Remove values below threshold
            filtered_df = df[df['y_true'] >= threshold]
            print(f"\nFiltering: keeping rows where y_true >= {threshold}")
        
        else:
            print(f"Error: Unknown filtering mode '{mode}'")
            return None
        
        print(f"Kept {len(filtered_df)} rows ({len(filtered_df)/len(df)*100:.1f}%)")
        print(f"Removed {len(df) - len(filtered_df)} rows")
        
        if len(filtered_df) == 0:
            print("Warning: No rows remain after filtering!")
            return None
        
        # Show filtered statistics
        print(f"\nFiltered y_true statistics:")
        print(f"  Range: {filtered_df['y_true'].min():.2f} to {filtered_df['y_true'].max():.2f}")
        print(f"  Mean: {filtered_df['y_true'].mean():.2f}")
        print(f"  Std: {filtered_df['y_true'].std():.2f}")
        print(f"  Median: {filtered_df['y_true'].median():.2f}")
        
        # Show y_pred statistics for comparison
        print(f"\nFiltered y_pred statistics:")
        print(f"  Range: {filtered_df['y_pred'].min():.2f} to {filtered_df['y_pred'].max():.2f}")
        print(f"  Mean: {filtered_df['y_pred'].mean():.2f}")
        print(f"  Std: {filtered_df['y_pred'].std():.2f}")
        print(f"  Median: {filtered_df['y_pred'].median():.2f}")
        
        # Calculate basic correlation
        correlation = filtered_df['y_true'].corr(filtered_df['y_pred'])
        print(f"\nCorrelation between y_true and y_pred: {correlation:.4f}")
        
        # Show distribution of signs
        true_positive = (filtered_df['y_true'] > 0).sum()
        true_negative = (filtered_df['y_true'] < 0).sum()
        pred_positive = (filtered_df['y_pred'] > 0).sum()
        pred_negative = (filtered_df['y_pred'] < 0).sum()
        
        print(f"\nSign distribution:")
        print(f"  y_true: {true_positive} positive, {true_negative} negative")
        print(f"  y_pred: {pred_positive} positive, {pred_negative} negative")
        
        # Calculate directional accuracy
        same_sign = ((filtered_df['y_true'] > 0) & (filtered_df['y_pred'] > 0)) | \
                   ((filtered_df['y_true'] < 0) & (filtered_df['y_pred'] < 0))
        directional_accuracy = same_sign.sum() / len(filtered_df)
        print(f"  Directional accuracy: {directional_accuracy:.4f} ({same_sign.sum()}/{len(filtered_df)})")
        
        # Save filtered data if output file specified
        if output_file:
            filtered_df.to_csv(output_file, sep='\t', index=False)
            print(f"\nSaved filtered data to {output_file}")
        
        return filtered_df
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Filter pairwise data to remove small perturbations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Remove small perturbations (keep only |y_true| > 5.0)
    python filter_pairwise.py --input diffs_raw.tsv --output diffs_filtered.tsv --threshold 5.0 --mode range
    
    # Remove small perturbations (alternative syntax)
    python filter_pairwise.py --input diffs_raw.tsv --output diffs_filtered.tsv --threshold 5.0 --mode abs
    
    # Keep only values below threshold (original behavior)
    python filter_pairwise.py --input diffs_raw.tsv --output diffs_filtered.tsv --threshold -5.0 --mode above
    
    # Keep only values above threshold
    python filter_pairwise.py --input diffs_raw.tsv --output diffs_filtered.tsv --threshold 5.0 --mode below
    
    # Preview without saving
    python filter_pairwise.py --input diffs_raw.tsv --threshold 5.0 --mode range --preview
        """
    )
    
    parser.add_argument("--input", "-i",
                        type=str, required=True,
                        help="Input TSV file with pairwise data")
    
    parser.add_argument("--output", "-o",
                        type=str, default=None,
                        help="Output TSV file for filtered data")
    
    parser.add_argument("--threshold", "-t",
                        type=float, required=True,
                        help="Threshold value for filtering y_true")
    
    parser.add_argument("--mode", "-m",
                        type=str, default="range",
                        choices=["range", "abs", "above", "below"],
                        help="Filtering mode: 'range' removes [-threshold, +threshold], 'abs' keeps |y_true| >= threshold, 'above' keeps y_true <= threshold, 'below' keeps y_true >= threshold")
    
    parser.add_argument("--preview", "-p",
                        action="store_true",
                        help="Preview filtering without saving output file")
    
    args = parser.parse_args()
    
    if args.preview:
        print("PREVIEW MODE - no output file will be saved")
        output_file = None
    else:
        output_file = args.output
        if not output_file:
            print("Error: --output is required unless using --preview")
            return 1
    
    # Filter the data
    result = filter_pairwise_data(
        input_file=args.input,
        output_file=output_file,
        threshold=args.threshold,
        mode=args.mode
    )
    
    if result is not None and not args.preview:
        print(f"\nFiltering complete!")
    elif args.preview:
        print(f"\nPreview complete. Use without --preview to save results.")
    else:
        print("Filtering failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())