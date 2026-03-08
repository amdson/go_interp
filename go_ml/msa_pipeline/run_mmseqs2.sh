#!/usr/bin/env bash
# run_mmseqs2.sh — Homolog search for all MSA datasets
#
# Usage (run from go_ml/msa_pipeline/):
#   bash run_mmseqs2.sh
#
# Prerequisites:
#   - MMseqs2 installed and on PATH (https://github.com/soedinglab/MMseqs2)
#   - UniRef database built in uniref_data/uniref_db_gpu
#     (build once with: mmseqs createdb <uniref.fasta> uniref_data/uniref_db &&
#                       mmseqs makepaddedseqdb uniref_data/uniref_db uniref_data/uniref_db_gpu)
#   - FASTA files for each dataset in ../../go_ml/gen_datasets/datasets/
#     (generate with build_fasta.ipynb if not present)
#
# Outputs:
#   {dataset}_aln.m8  — tab-separated alignment hits (input to build_homologues.ipynb)
#
# Settings: 2 search iterations, GPU-accelerated, sequence+alignment output columns

set -euo pipefail

DATASETS=("csa" "llps" "elms" "ip_domain")
FASTA_DIR="../../go_ml/gen_datasets/datasets"
UNIREF_DB="uniref_data/uniref_db_gpu"
QUERY_DB_DIR="query_db"
RESULTS_DB_DIR="results_db"
TMP_DIR="tmp"
FORMAT="query,target,fident,pident,nident,qlen,tlen,qstart,qend,tstart,tend,evalue,bits,qseq,tseq"

mkdir -p "$QUERY_DB_DIR" "$RESULTS_DB_DIR" "$TMP_DIR"

for DATASET in "${DATASETS[@]}"; do
    echo "=== Processing $DATASET ==="
    FASTA="$FASTA_DIR/${DATASET}.fasta"
    QUERY_DB="$QUERY_DB_DIR/${DATASET}_db"
    RESULT_DB="$RESULTS_DB_DIR/${DATASET}_result_db"
    M8_OUT="${DATASET}_aln.m8"

    if [ -f "$M8_OUT" ]; then
        echo "  Skipping: $M8_OUT already exists"
        continue
    fi

    echo "  Step 1: Creating query DB from $FASTA"
    mmseqs createdb "$FASTA" "$QUERY_DB"

    echo "  Step 2: Searching against UniRef"
    mmseqs search "$QUERY_DB" "$UNIREF_DB" "$RESULT_DB" "$TMP_DIR" \
        --gpu 1 --num-iterations 2

    echo "  Step 3: Converting results to m8 format"
    mmseqs convertalis "$QUERY_DB" "$UNIREF_DB" "$RESULT_DB" "$M8_OUT" \
        --format-output "$FORMAT"

    echo "  Done: $M8_OUT"
done

echo "All datasets complete."
