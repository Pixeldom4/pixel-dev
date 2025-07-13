#!/bin/bash
# run_pipeline.bash
INPUT_TSV="$1"
REFERENCE="$2"
EXPERIMENT_NAME="$3"
OUTPUT_DIR="../experiments/$EXPERIMENT_NAME/"

echo "Starting TM analysis pipeline for experiment: $EXPERIMENT_NAME"

mkdir -p "$OUTPUT_DIR"

bash exclude_reference.bash "$INPUT_TSV" "$REFERENCE" "$OUTPUT_DIR"
NO_REF_FILE=$(basename "$INPUT_TSV" .tsv)_no_ref.tsv

bash generate_pairwise.bash "$OUTPUT_DIR$NO_REF_FILE" "$OUTPUT_DIR"
PAIRWISE_FILE=$(basename "$INPUT_TSV" .tsv)_no_ref_pairwise.tsv

bash create_matrix.bash "$OUTPUT_DIR$PAIRWISE_FILE" "$REFERENCE" "$OUTPUT_DIR"
MATRIX_FILE=$(basename "$INPUT_TSV" .tsv)_no_ref_pairwise_matrix_${REFERENCE//\//_}.tsv

bash entropy_analysis.bash "$OUTPUT_DIR$MATRIX_FILE" "$OUTPUT_DIR"

echo "Pipeline completed. Results in $OUTPUT_DIR"