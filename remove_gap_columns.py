#!/usr/bin/env python3

import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sys

def remove_gap_columns(input_file, output_file):
    """
    Remove columns from MSA that contain any gap characters.
    Only keep columns with 100% amino acid coverage (no gaps).
    """
    
    # Read all sequences
    print(f"Reading alignment from {input_file}...")
    records = list(SeqIO.parse(input_file, "fasta"))
    
    if not records:
        print("Error: No sequences found in input file")
        sys.exit(1)
    
    print(f"Found {len(records)} sequences")
    
    # Get alignment length
    alignment_length = len(records[0].seq)
    print(f"Original alignment length: {alignment_length}")
    
    # Check all sequences have same length
    for i, record in enumerate(records):
        if len(record.seq) != alignment_length:
            print(f"Error: Sequence {record.id} has different length ({len(record.seq)}) than first sequence ({alignment_length})")
            sys.exit(1)
    
    # Find columns without gaps
    good_columns = []
    gap_chars = {'-', '.', 'X', 'x'}  # Common gap characters
    
    print("Analyzing columns for gaps...")
    for col_idx in range(alignment_length):
        # Check if this column has any gaps
        has_gap = False
        for record in records:
            if record.seq[col_idx] in gap_chars:
                has_gap = True
                break
        
        if not has_gap:
            good_columns.append(col_idx)
    
    print(f"Columns without gaps: {len(good_columns)}")
    print(f"Columns with gaps removed: {alignment_length - len(good_columns)}")
    
    if len(good_columns) == 0:
        print("Error: No columns without gaps found!")
        sys.exit(1)
    
    # Create new sequences with only good columns
    new_records = []
    for record in records:
        new_seq = ''.join([record.seq[i] for i in good_columns])
        new_record = SeqRecord(
            Seq(new_seq),
            id=record.id,
            description=record.description
        )
        new_records.append(new_record)
    
    # Write output
    print(f"Writing gap-free alignment to {output_file}...")
    SeqIO.write(new_records, output_file, "fasta")
    
    print(f"Success! New alignment length: {len(good_columns)}")
    print(f"Removed {alignment_length - len(good_columns)} columns containing gaps")

def main():
    parser = argparse.ArgumentParser(
        description="Remove columns containing gaps from multiple sequence alignment",
        epilog="Example: python remove_gap_columns.py --input alignment.afa --output alignment.nogaps.afa"
    )
    
    parser.add_argument("--input", "-i", 
                        type=str, 
                        required=True,
                        help="Input MSA file in FASTA format")
    
    parser.add_argument("--output", "-o", 
                        type=str, 
                        required=True,
                        help="Output gap-free MSA file")
    
    args = parser.parse_args()
    
    # Check input file exists
    try:
        with open(args.input, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Remove gap columns
    remove_gap_columns(args.input, args.output)

if __name__ == "__main__":
    main()