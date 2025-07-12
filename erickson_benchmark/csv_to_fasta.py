#!/usr/bin/env python3
"""
Convert CSV file with protein sequences to FASTA format.
Supports flexible column mapping and various output options.
"""

import argparse
import pandas as pd
import os
import sys
from pathlib import Path

def read_csv_file(csv_file, delimiter=','):
    """Read CSV file and return DataFrame"""
    try:
        # Try to detect delimiter if not specified
        if delimiter == 'auto':
            # Read first line to detect delimiter
            with open(csv_file, 'r') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    delimiter = '\t'
                elif ',' in first_line:
                    delimiter = ','
                elif ';' in first_line:
                    delimiter = ';'
                else:
                    delimiter = ','
            print(f"Auto-detected delimiter: '{delimiter}'")
        
        df = pd.read_csv(csv_file, sep=delimiter)
        print(f"Successfully read {len(df)} rows from {csv_file}")
        print(f"Available columns: {list(df.columns)}")
        return df
    
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return None

def validate_columns(df, id_column, sequence_column):
    """Validate that required columns exist in DataFrame"""
    missing_columns = []
    
    if id_column not in df.columns:
        missing_columns.append(id_column)
    
    if sequence_column not in df.columns:
        missing_columns.append(sequence_column)
    
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    return True

def clean_sequence(sequence):
    """Clean and validate protein sequence"""
    if pd.isna(sequence):
        return ""
    
    # Convert to string and remove whitespace
    seq_str = str(sequence).strip()
    
    # Remove any non-amino acid characters (optional - can be made configurable)
    # Keep only standard amino acid letters
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY*-')
    cleaned_seq = ''.join(c.upper() for c in seq_str if c.upper() in valid_chars)
    
    return cleaned_seq

def create_fasta_header(row, id_column, description_columns=None, custom_format=None):
    """Create FASTA header from row data"""
    # Get the main identifier
    main_id = str(row[id_column]).strip()
    
    if custom_format:
        # Use custom format string
        try:
            header = custom_format.format(**row.to_dict())
        except KeyError as e:
            print(f"Warning: Column {e} not found for custom format. Using default format.")
            header = main_id
    elif description_columns:
        # Add description columns
        descriptions = []
        for col in description_columns:
            if col in row.index and pd.notna(row[col]):
                descriptions.append(f"{col}={row[col]}")
        
        if descriptions:
            header = f"{main_id} {' '.join(descriptions)}"
        else:
            header = main_id
    else:
        # Simple header with just the ID
        header = main_id
    
    return header

def convert_csv_to_fasta(df, id_column, sequence_column, output_file, 
                        description_columns=None, custom_format=None, 
                        min_length=0, max_length=None, line_width=80,
                        skip_empty=True, add_counter=False):
    """Convert DataFrame to FASTA format"""
    
    sequences_written = 0
    sequences_skipped = 0
    
    try:
        with open(output_file, 'w') as f:
            for idx, row in df.iterrows():
                # Get and clean sequence
                raw_sequence = row[sequence_column]
                sequence = clean_sequence(raw_sequence)
                
                # Skip empty sequences if requested
                if skip_empty and not sequence:
                    sequences_skipped += 1
                    continue
                
                # Apply length filters
                if len(sequence) < min_length:
                    sequences_skipped += 1
                    continue
                
                if max_length and len(sequence) > max_length:
                    sequences_skipped += 1
                    continue
                
                # Create header
                header = create_fasta_header(row, id_column, description_columns, custom_format)
                
                # Add counter if requested
                if add_counter:
                    header = f"{sequences_written + 1}_{header}"
                
                # Write FASTA entry
                f.write(f">{header}\n")
                
                # Write sequence with line wrapping
                if line_width > 0:
                    for i in range(0, len(sequence), line_width):
                        f.write(f"{sequence[i:i+line_width]}\n")
                else:
                    f.write(f"{sequence}\n")
                
                sequences_written += 1
        
        print(f"Successfully wrote {sequences_written} sequences to {output_file}")
        if sequences_skipped > 0:
            print(f"Skipped {sequences_skipped} sequences (empty, too short, or too long)")
        
        return True
    
    except Exception as e:
        print(f"Error writing FASTA file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV file with protein sequences to FASTA format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion (uses first column as ID, second as sequence)
    python csv_to_fasta.py input.csv output.fasta
    
    # Specify columns by name
    python csv_to_fasta.py input.csv output.fasta --id_column "Enzyme ID" --sequence_column "Protein Sequence"
    
    # Specify columns by index (0-based)
    python csv_to_fasta.py input.csv output.fasta --id_column_index 2 --sequence_column_index 3
    
    # With description columns
    python csv_to_fasta.py input.csv output.fasta --description "Species" "Function"
    
    # Custom header format
    python csv_to_fasta.py input.csv output.fasta --custom_format "{ID}|{Species}|{Function}"
    
    # With filtering
    python csv_to_fasta.py input.csv output.fasta --min_length 50 --max_length 1000
    
    # Tab-separated input
    python csv_to_fasta.py input.tsv output.fasta --delimiter tab
    
    # Preview mode
    python csv_to_fasta.py input.csv output.fasta --preview
        """
    )
    
    # Required arguments
    parser.add_argument('input_csv', 
                       help='Input CSV file path')
    parser.add_argument('output_fasta', 
                       help='Output FASTA file path')
    
    # Column specification
    parser.add_argument('-i', '--id_column', 
                       default=None,
                       help='Column name containing sequence identifiers (default: first column)')
    parser.add_argument('-s', '--sequence_column', 
                       default=None,
                       help='Column name containing protein sequences (default: second column)')
    parser.add_argument('--id_column_index', 
                       type=int,
                       default=0,
                       help='Column index for sequence identifiers (0-based, default: 0)')
    parser.add_argument('--sequence_column_index', 
                       type=int,
                       default=1,
                       help='Column index for protein sequences (0-based, default: 1)')
    parser.add_argument('-d', '--description', 
                       nargs='*', 
                       dest='description_columns',
                       help='Additional columns to include in FASTA headers')
    
    # File format options
    parser.add_argument('--delimiter', 
                       choices=['comma', 'tab', 'semicolon', 'auto'], 
                       default='comma',
                       help='CSV delimiter (default: comma)')
    
    # Header formatting
    parser.add_argument('--custom_format', 
                       help='Custom header format string (e.g., "{ID}|{Species}|{Function}")')
    parser.add_argument('--add_counter', 
                       action='store_true',
                       help='Add sequential counter to each header')
    
    # Sequence filtering
    parser.add_argument('--min_length', 
                       type=int, 
                       default=0,
                       help='Minimum sequence length (default: 0)')
    parser.add_argument('--max_length', 
                       type=int,
                       help='Maximum sequence length (no limit by default)')
    parser.add_argument('--include_empty', 
                       action='store_true',
                       help='Include sequences that are empty or contain only invalid characters')
    
    # Output formatting
    parser.add_argument('--line_width', 
                       type=int, 
                       default=80,
                       help='Number of characters per line in FASTA output (default: 80, 0 for no wrapping)')
    
    # Utility options
    parser.add_argument('--preview', 
                       action='store_true',
                       help='Preview first 5 entries without writing output file')
    parser.add_argument('--show_stats', 
                       action='store_true',
                       help='Show statistics about sequences')
    
    args = parser.parse_args()
    
    # Convert delimiter argument to actual character
    delimiter_map = {
        'comma': ',',
        'tab': '\t',
        'semicolon': ';',
        'auto': 'auto'
    }
    delimiter = delimiter_map[args.delimiter]
    
    # Check if input file exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file '{args.input_csv}' does not exist")
        return 1
    
    # Read CSV file
    print(f"Reading CSV file: {args.input_csv}")
    df = read_csv_file(args.input_csv, delimiter)
    if df is None:
        return 1
    
    # Determine column names to use
    if args.id_column is None:
        # Use column index
        if args.id_column_index >= len(df.columns):
            print(f"Error: ID column index {args.id_column_index} is out of range. File has {len(df.columns)} columns.")
            return 1
        id_column = df.columns[args.id_column_index]
        print(f"Using column index {args.id_column_index} ('{id_column}') for sequence IDs")
    else:
        id_column = args.id_column
        print(f"Using column name '{id_column}' for sequence IDs")
    
    if args.sequence_column is None:
        # Use column index
        if args.sequence_column_index >= len(df.columns):
            print(f"Error: Sequence column index {args.sequence_column_index} is out of range. File has {len(df.columns)} columns.")
            return 1
        sequence_column = df.columns[args.sequence_column_index]
        print(f"Using column index {args.sequence_column_index} ('{sequence_column}') for sequences")
    else:
        sequence_column = args.sequence_column
        print(f"Using column name '{sequence_column}' for sequences")
    
    # Validate columns
    if not validate_columns(df, id_column, sequence_column):
        return 1
    
    # Show statistics if requested
    if args.show_stats:
        print(f"\nSequence Statistics:")
        sequences = df[sequence_column].apply(clean_sequence)
        non_empty = sequences[sequences.str.len() > 0]
        
        if len(non_empty) > 0:
            print(f"Total sequences: {len(df)}")
            print(f"Non-empty sequences: {len(non_empty)}")
            print(f"Average length: {non_empty.str.len().mean():.1f}")
            print(f"Min length: {non_empty.str.len().min()}")
            print(f"Max length: {non_empty.str.len().max()}")
            print(f"Median length: {non_empty.str.len().median():.1f}")
        else:
            print("No valid sequences found!")
    
    # Preview mode
    if args.preview:
        print(f"\nPreview of first 5 entries:")
        preview_df = df.head(5)
        for idx, row in preview_df.iterrows():
            sequence = clean_sequence(row[sequence_column])
            header = create_fasta_header(row, id_column, 
                                       args.description_columns, args.custom_format)
            if args.add_counter:
                header = f"{idx + 1}_{header}"
            
            print(f">{header}")
            if len(sequence) > 60:
                print(f"{sequence[:60]}... (length: {len(sequence)})")
            else:
                print(f"{sequence}")
            print()
        
        return 0
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_fasta)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Convert to FASTA
    print(f"\nConverting to FASTA format...")
    success = convert_csv_to_fasta(
        df, 
        id_column, 
        sequence_column, 
        args.output_fasta,
        description_columns=args.description_columns,
        custom_format=args.custom_format,
        min_length=args.min_length,
        max_length=args.max_length,
        line_width=args.line_width,
        skip_empty=not args.include_empty,
        add_counter=args.add_counter
    )
    
    if success:
        print(f"\nConversion completed successfully!")
        print(f"Output file: {args.output_fasta}")
        
        # Show file size
        file_size = os.path.getsize(args.output_fasta)
        if file_size > 1024 * 1024:
            print(f"File size: {file_size / (1024 * 1024):.1f} MB")
        elif file_size > 1024:
            print(f"File size: {file_size / 1024:.1f} KB")
        else:
            print(f"File size: {file_size} bytes")
        
        return 0
    else:
        print("Conversion failed!")
        return 1

if __name__ == "__main__":
    exit(main())