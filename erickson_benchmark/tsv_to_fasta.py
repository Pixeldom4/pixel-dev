import pandas as pd
import argparse

def tsv_to_fasta(tsv_file, fasta_file, accession_col=0, sequence_col=1, wrap_length=80):
    """
    Convert TSV file to FASTA format
    
    Args:
        tsv_file: Path to input TSV file
        fasta_file: Path to output FASTA file
        accession_col: Column index for accession (0-based, default: 0)
        sequence_col: Column index for sequence (0-based, default: 1)
        wrap_length: Number of characters per line for sequence (default: 80, 0 = no wrapping)
    """
    try:
        # Read TSV file without header
        df = pd.read_csv(tsv_file, sep='\t', header=None)
        
        print(f"File contains {len(df.columns)} columns and {len(df)} rows")
        print(f"Using column {accession_col} for accessions and column {sequence_col} for sequences")
        
        # Check if specified columns exist
        if accession_col >= len(df.columns):
            raise ValueError(f"Accession column {accession_col} does not exist. File has {len(df.columns)} columns (0-{len(df.columns)-1})")
        if sequence_col >= len(df.columns):
            raise ValueError(f"Sequence column {sequence_col} does not exist. File has {len(df.columns)} columns (0-{len(df.columns)-1})")
        
        sequences_written = 0
        skipped_rows = 0
        
        with open(fasta_file, 'w') as f:
            for idx, row in df.iterrows():
                accession = str(row.iloc[accession_col]).strip() if pd.notna(row.iloc[accession_col]) else ""
                sequence = str(row.iloc[sequence_col]).strip() if pd.notna(row.iloc[sequence_col]) else ""
                
                # Skip rows with missing data
                if not accession or not sequence or accession == 'nan' or sequence == 'nan':
                    print(f"Skipping row {idx + 1}: missing accession or sequence data")
                    skipped_rows += 1
                    continue
                
                # Write FASTA header
                f.write(f">{accession}\n")
                
                # Write sequence (with optional line wrapping)
                if wrap_length > 0:
                    # Wrap sequence to specified length
                    for i in range(0, len(sequence), wrap_length):
                        f.write(sequence[i:i+wrap_length] + '\n')
                else:
                    # No wrapping - single line
                    f.write(sequence + '\n')
                
                sequences_written += 1
        
        print(f"Successfully converted {sequences_written} sequences to FASTA format")
        if skipped_rows > 0:
            print(f"Skipped {skipped_rows} rows due to missing data")
        print(f"Output saved to: {fasta_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{tsv_file}'")
    except Exception as e:
        print(f"Error converting TSV to FASTA: {e}")

def preview_tsv(tsv_file, num_rows=5):
    """
    Preview the TSV file to help users identify column indices
    """
    try:
        df = pd.read_csv(tsv_file, sep='\t', header=None, nrows=num_rows)
        print(f"Preview of {tsv_file} (first {num_rows} rows):")
        print(f"File has {len(df.columns)} columns (indices 0-{len(df.columns)-1})")
        print("-" * 80)
        
        for idx, row in df.iterrows():
            print(f"Row {idx + 1}:")
            for col_idx, value in enumerate(row):
                value_str = str(value)[:50] + ('...' if len(str(value)) > 50 else '')
                print(f"  Column {col_idx}: {value_str}")
            print()
            
    except Exception as e:
        print(f"Error previewing file: {e}")

def validate_fasta(fasta_file, sample_size=5):
    """
    Quick validation of the generated FASTA file
    """
    try:
        print(f"\nValidating FASTA file: {fasta_file}")
        
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
        
        headers = [line for line in lines if line.startswith('>')]
        sequences = [line for line in lines if not line.startswith('>') and line.strip()]
        
        print(f"Total headers found: {len(headers)}")
        print(f"Total sequence lines found: {len(sequences)}")
        
        # Show sample entries
        print(f"\nFirst {sample_size} entries:")
        header_count = 0
        for i, line in enumerate(lines[:sample_size*3]):  # Show more lines to capture sequences
            if line.startswith('>'):
                header_count += 1
                if header_count <= sample_size:
                    print(f"Header {header_count}: {line.strip()}")
            elif line.strip() and header_count <= sample_size:
                seq_preview = line.strip()[:50] + ('...' if len(line.strip()) > 50 else '')
                print(f"Sequence: {seq_preview}")
                print()
        
    except Exception as e:
        print(f"Error validating FASTA file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert TSV file to FASTA format with flexible column selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python script.py input.tsv output.fasta
  python script.py input.tsv output.fasta --accession-col 2 --sequence-col 0
  python script.py input.tsv output.fasta --preview
  python script.py input.tsv output.fasta --wrap 60 --validate
        '''
    )
    
    parser.add_argument('tsv_file', help='Input TSV file')
    parser.add_argument('fasta_file', help='Output FASTA file')
    parser.add_argument('--accession-col', type=int, default=0, 
                       help='Column index for accessions (0-based, default: 0)')
    parser.add_argument('--sequence-col', type=int, default=1, 
                       help='Column index for sequences (0-based, default: 1)')
    parser.add_argument('--wrap', type=int, default=80, 
                       help='Characters per line for sequences (default: 80, use 0 for no wrapping)')
    parser.add_argument('--preview', action='store_true', 
                       help='Preview the TSV file structure before conversion')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate the output FASTA file after conversion')
    
    args = parser.parse_args()
    
    # Validate column indices
    if args.accession_col < 0 or args.sequence_col < 0:
        print("Error: Column indices must be non-negative")
        return
    
    # Preview file if requested
    if args.preview:
        preview_tsv(args.tsv_file)
        print("\nProceed with conversion? (y/n): ", end='')
        if input().lower() not in ['y', 'yes']:
            print("Conversion cancelled")
            return
    
    # Convert TSV to FASTA
    tsv_to_fasta(args.tsv_file, args.fasta_file, args.accession_col, args.sequence_col, args.wrap)
    
    # Optional validation
    if args.validate:
        validate_fasta(args.fasta_file)

# Simple function for direct use
def simple_tsv_to_fasta(tsv_file, fasta_file='output.fasta', accession_col=0, sequence_col=1):
    """
    Simple function to convert TSV to FASTA - can be called directly
    """
    tsv_to_fasta(tsv_file, fasta_file, accession_col, sequence_col)
    return fasta_file

if __name__ == "__main__":
    main()