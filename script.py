import csv
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import argparse
from collections import defaultdict
import time

def read_csv_sequences(csv_file):
    """Read protein sequences from CSV file"""
    sequences = {}
    try:
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            enzyme_id = row['Enzyme ID']
            protein_seq = row['Protein Sequence']
            sequences[enzyme_id] = protein_seq.strip() if pd.notna(protein_seq) else ""
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}
    return sequences

def read_fasta_sequences(fasta_file):
    """Read sequences from FASTA file"""
    sequences = {}
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences[record.id] = str(record.seq)
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return {}
    return sequences

def create_kmer_index(sequences, k=6):
    """Create k-mer index for fast substring search"""
    kmer_index = defaultdict(set)
    
    for seq_id, sequence in sequences.items():
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmer_index[kmer].add(seq_id)
    
    return kmer_index

def fast_longest_common_substring(seq1, seq2, min_length=6):
    """
    Fast LCS using rolling hash - much faster than DP approach
    """
    if len(seq1) < min_length or len(seq2) < min_length:
        return "", 0, -1, -1
    
    max_length = 0
    best_substring = ""
    best_pos1 = -1
    best_pos2 = -1
    
    # Start with longer possible substrings and work down
    max_possible = min(len(seq1), len(seq2))
    
    for length in range(min(max_possible, 50), min_length - 1, -1):  # Cap at 50 for performance
        found = False
        
        # Create set of all substrings of this length from seq1
        substrings_seq1 = {}
        for i in range(len(seq1) - length + 1):
            substring = seq1[i:i+length]
            if substring not in substrings_seq1:
                substrings_seq1[substring] = i
        
        # Check if any appear in seq2
        for i in range(len(seq2) - length + 1):
            substring = seq2[i:i+length]
            if substring in substrings_seq1:
                max_length = length
                best_substring = substring
                best_pos1 = substrings_seq1[substring]
                best_pos2 = i
                found = True
                break
        
        if found:
            break
    
    return best_substring, max_length, best_pos1, best_pos2

def calculate_kmer_similarity(seq1, seq2, k=6):
    """Calculate similarity based on shared k-mers (much faster)"""
    if len(seq1) < k or len(seq2) < k:
        return 0.0
    
    kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))
    
    if not kmers1 or not kmers2:
        return 0.0
    
    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    
    return (intersection / union) * 100 if union > 0 else 0.0

def optimized_robust_search(csv_sequences, fasta_sequences, min_overlap=6, similarity_threshold=30.0, kmer_size=6):
    """
    Optimized robust search using k-mer indexing and fast algorithms
    """
    print("Creating k-mer index for FASTA sequences...")
    fasta_kmer_index = create_kmer_index(fasta_sequences, kmer_size)
    
    results = []
    total_sequences = len(csv_sequences)
    
    for idx, (enzyme_id, csv_seq) in enumerate(csv_sequences.items()):
        if idx % 10 == 0:  # Progress indicator
            print(f"Processing sequence {idx + 1}/{total_sequences}")
        
        if not csv_seq or len(csv_seq) < min_overlap:
            results.append({
                'Enzyme_ID': enzyme_id,
                'CSV_Sequence': 'Too short or empty',
                'CSV_Length': len(csv_seq) if csv_seq else 0,
                'Match_Type': 'No Match',
                'Best_Match_Details': 'Sequence too short',
                'Number_of_Matches': 0,
                'Top_Similarity': 0.0,
                'Top_LCS_Length': 0,
                'All_Matches': 'None'
            })
            continue
        
        # Step 1: Find candidate sequences using k-mer overlap
        candidate_sequences = set()
        csv_kmers = set(csv_seq[i:i+kmer_size] for i in range(len(csv_seq) - kmer_size + 1))
        
        for kmer in csv_kmers:
            if kmer in fasta_kmer_index:
                candidate_sequences.update(fasta_kmer_index[kmer])
        
        if not candidate_sequences:
            results.append({
                'Enzyme_ID': enzyme_id,
                'CSV_Sequence': csv_seq[:50] + '...' if len(csv_seq) > 50 else csv_seq,
                'CSV_Length': len(csv_seq),
                'Match_Type': 'No Match',
                'Best_Match_Details': 'No k-mer overlap found',
                'Number_of_Matches': 0,
                'Top_Similarity': 0.0,
                'Top_LCS_Length': 0,
                'All_Matches': 'None'
            })
            continue
        
        # Step 2: Analyze only candidate sequences
        best_matches = []
        
        for fasta_id in candidate_sequences:
            fasta_seq = fasta_sequences[fasta_id]
            
            # Quick exact match check
            exact_match_type = None
            if csv_seq == fasta_seq:
                exact_match_type = "Exact_Match"
            elif csv_seq in fasta_seq:
                exact_match_type = "CSV_in_FASTA"
            elif fasta_seq in csv_seq:
                exact_match_type = "FASTA_in_CSV"
            
            # Fast similarity calculation
            kmer_similarity = calculate_kmer_similarity(csv_seq, fasta_seq, kmer_size)
            
            # Only do expensive LCS calculation if k-mer similarity is promising
            lcs_length = 0
            lcs = ""
            if kmer_similarity >= similarity_threshold * 0.5 or exact_match_type:  # Lower threshold for LCS
                lcs, lcs_length, _, _ = fast_longest_common_substring(csv_seq, fasta_seq, min_overlap)
            
            # Determine if this is a significant match
            is_significant = (
                exact_match_type or 
                lcs_length >= min_overlap or 
                kmer_similarity >= similarity_threshold
            )
            
            if is_significant:
                best_matches.append({
                    'fasta_id': fasta_id,
                    'exact_match_type': exact_match_type,
                    'lcs_length': lcs_length,
                    'kmer_similarity': round(kmer_similarity, 2),
                    'longest_common_substring': lcs[:20] + '...' if len(lcs) > 20 else lcs
                })
        
        # Sort by k-mer similarity and LCS length
        best_matches.sort(key=lambda x: (x['kmer_similarity'], x['lcs_length']), reverse=True)
        
        # Generate result
        if not best_matches:
            match_type = 'No Match'
            match_details = 'No significant matches found'
        else:
            top_match = best_matches[0]
            if top_match['exact_match_type']:
                match_type = f"Exact ({top_match['exact_match_type']})"
            elif top_match['kmer_similarity'] >= similarity_threshold:
                match_type = f"High Similarity ({top_match['kmer_similarity']}%)"
            else:
                match_type = f"Partial Match (LCS: {top_match['lcs_length']}aa)"
            
            match_details = f"{top_match['fasta_id']} (Sim: {top_match['kmer_similarity']}%, LCS: {top_match['lcs_length']}aa)"
        
        results.append({
            'Enzyme_ID': enzyme_id,
            'CSV_Sequence': csv_seq[:50] + '...' if len(csv_seq) > 50 else csv_seq,
            'CSV_Length': len(csv_seq),
            'Match_Type': match_type,
            'Best_Match_Details': match_details,
            'Number_of_Matches': len(best_matches),
            'Top_Similarity': best_matches[0]['kmer_similarity'] if best_matches else 0.0,
            'Top_LCS_Length': best_matches[0]['lcs_length'] if best_matches else 0,
            'All_Matches': '; '.join([f"{m['fasta_id']}({m['kmer_similarity']}%)" 
                                    for m in best_matches[:5]]) if best_matches else 'None'
        })
    
    return results

def exact_match_search(csv_sequences, fasta_sequences):
    """Fast exact match search"""
    # Create reverse index for FASTA sequences
    fasta_by_sequence = {}
    for fasta_id, fasta_seq in fasta_sequences.items():
        if fasta_seq not in fasta_by_sequence:
            fasta_by_sequence[fasta_seq] = []
        fasta_by_sequence[fasta_seq].append(fasta_id)
    
    results = []
    for enzyme_id, csv_seq in csv_sequences.items():
        matches = fasta_by_sequence.get(csv_seq, [])
        
        results.append({
            'Enzyme_ID': enzyme_id,
            'CSV_Sequence': csv_seq,
            'Match_Type': 'Exact' if matches else 'No Match',
            'Matched_FASTA_IDs': ', '.join(matches) if matches else 'None',
            'Number_of_Matches': len(matches)
        })
    
    return results

def save_results(results, output_file):
    """Save search results to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Optimized protein sequence search')
    parser.add_argument('csv_file', help='Path to CSV file with Enzyme ID and Protein Sequence columns')
    parser.add_argument('fasta_file', help='Path to FASTA file with protein sequences')
    parser.add_argument('output_file', help='Path to output CSV file for results')
    parser.add_argument('--search_type', choices=['exact', 'robust'], default='robust',
                       help='Type of search: exact or robust (default)')
    parser.add_argument('--min_overlap', type=int, default=6,
                       help='Minimum overlap length for matches (default: 6)')
    parser.add_argument('--similarity_threshold', type=float, default=30.0,
                       help='Minimum similarity percentage (default: 30.0)')
    parser.add_argument('--kmer_size', type=int, default=6,
                       help='K-mer size for indexing (default: 6)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("Reading CSV sequences...")
    csv_sequences = read_csv_sequences(args.csv_file)
    print(f"Found {len(csv_sequences)} sequences in CSV")
    
    print("Reading FASTA sequences...")
    fasta_sequences = read_fasta_sequences(args.fasta_file)
    print(f"Found {len(fasta_sequences)} sequences in FASTA")
    
    print(f"Performing {args.search_type} search...")
    
    if args.search_type == 'exact':
        results = exact_match_search(csv_sequences, fasta_sequences)
    else:  # robust
        results = optimized_robust_search(csv_sequences, fasta_sequences, 
                                        args.min_overlap, args.similarity_threshold, args.kmer_size)
    
    save_results(results, args.output_file)
    
    # Print summary
    total_searches = len(results)
    matches = sum(1 for r in results if r['Number_of_Matches'] > 0)
    elapsed_time = time.time() - start_time
    
    print(f"\nSummary:")
    print(f"Total sequences searched: {total_searches}")
    print(f"Sequences with matches: {matches}")
    print(f"Sequences without matches: {total_searches - matches}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    if args.search_type == 'robust' and matches > 0:
        avg_similarity = sum(r.get('Top_Similarity', 0) for r in results if r['Number_of_Matches'] > 0) / matches
        print(f"Average similarity of matches: {avg_similarity:.2f}%")

if __name__ == "__main__":
    main()