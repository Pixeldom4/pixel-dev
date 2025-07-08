import pandas as pd
import argparse

def tsv_to_fasta(tsv_file, fasta_file, wrap_length=80):
    """
    Convert TSV file to FASTA format
    
    Args:
        tsv_file: Path to input TSV file (no header, columns: accession, sequence)
        fasta_file: Path to output FASTA file
        wrap_length: Number of characters per line for sequence (default: 80, 0 = no wrapping)
    """
    try:
        # Read TSV file without header
        df = pd.read_csv(tsv_file, sep='\t', header=None, names=['accession', 'sequence'])
        
        sequences_written = 0
        
        with open(fasta_file, 'w') as f:
            for _, row in df.iterrows():
                accession = str(row['accession']).strip()
                sequence = str(row['sequence']).strip()
                
                # Skip rows with missing data
                if pd.isna(row['accession']) or pd.isna(row['sequence']) or not sequence:
                    print(f"Skipping row with missing data: {accession}")
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
        print(f"Output saved to: {fasta_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{tsv_file}'")
    except Exception as e:
        print(f"Error converting TSV to FASTA: {e}")

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
    parser = argparse.ArgumentParser(description='Convert TSV file to FASTA format')
    parser.add_argument('tsv_file', help='Input TSV file (no header, columns: accession, sequence)')
    parser.add_argument('fasta_file', help='Output FASTA file')
    parser.add_argument('--wrap', type=int, default=80, 
                       help='Characters per line for sequences (default: 80, use 0 for no wrapping)')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate the output FASTA file after conversion')
    
    args = parser.parse_args()
    
    # Convert TSV to FASTA
    tsv_to_fasta(args.tsv_file, args.fasta_file, args.wrap)
    
    # Optional validation
    if args.validate:
        validate_fasta(args.fasta_file)

# Simple function for direct use
def simple_tsv_to_fasta(tsv_file, fasta_file='output.fasta'):
    """
    Simple function to convert TSV to FASTA - can be called directly
    """
    tsv_to_fasta(tsv_file, fasta_file)
    return fasta_file

if __name__ == "__main__":
    main()

# Example usage:
# python tsv_to_fasta.py input.tsv output.fasta
# python tsv_to_fasta.py input.tsv output.fasta --wrap 60 --validate