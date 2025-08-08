QUERY_NAME := csa

QUERY_FASTA := /home/andrew/GO_interp/go_ml/gen_datasets/datasets/csa.fasta

UNIPROT_FASTA := uniprot_sprot.fasta
PROT_DB_NAME := sprot_db
PROT_DB_PADDED := sprot_db_gpu
UNIREF_DB := uniref_data/uniref_db

# Directory variables
QUERY_DB_DIR := query_db
RESULTS_DB_DIR := results_db
TMP_DIR := tmp

# Output file
M8_OUTPUT := $(QUERY_NAME)_aln.m8

# ================================================================

# Phony targets don't represent a file, they are just a name for a set of commands.
.PHONY: all clean sprot_db_prep query_db_prep

# The main target. It depends on the final output file.
all: $(M8_OUTPUT)

# --- Step 1: Prepare the main uniprot database ---
# This target creates the GPU-optimized database and cleans up intermediate files.
$(PROT_DB_PADDED): $(UNIPROT_FASTA)
	@echo "--- Step 1: Converting Uniprot FASTA to GPU database ---"
	mmseqs createdb $(UNIPROT_FASTA) $(PROT_DB_NAME)
	mmseqs makepaddedseqdb $(PROT_DB_NAME) $(PROT_DB_PADDED)
	# mmseqs rmdb $(PROT_DB_NAME) # Keeping the original db in case you need it later.

# --- Step 2: Prepare the query database ---
# This target creates the query database from the user's fasta file.
$(QUERY_DB_DIR)/$(QUERY_NAME)_db: $(QUERY_FASTA)
	@echo "--- Step 2: Converting query FASTA to MMSeqs2 database ---"
	mkdir -p $(QUERY_DB_DIR)
	mmseqs createdb $(QUERY_FASTA) $(QUERY_DB_DIR)/$(QUERY_NAME)_db

# --- Step 3: Run the search and convert results ---
# This target runs the search and then converts the results to a readable format.
$(M8_OUTPUT): $(PROT_DB_PADDED) $(QUERY_DB_DIR)/$(QUERY_NAME)_db
	@echo "--- Step 3: Running MMSeqs2 search and converting results ---"
	mkdir -p $(RESULTS_DB_DIR)
	mmseqs search $(QUERY_DB_DIR)/$(QUERY_NAME)_db $(UNIREF_DB) $(RESULTS_DB_DIR)/$(QUERY_NAME)_result_db $(TMP_DIR) --gpu 1 --num-iterations 2
	mmseqs convertalis $(QUERY_DB_DIR)/$(QUERY_NAME)_db $(UNIREF_DB) $(RESULTS_DB_DIR)/$(QUERY_NAME)_result_db $(M8_OUTPUT) --format-output "query,target,fident,pident,nident,qlen,tlen,qstart,qend,tstart,tend,evalue,bits,qaln,taln"

# --- Clean up ---
# This target removes all generated files and databases.
clean:
	@echo "--- Cleaning up all generated files and directories ---"
	rm -rf $(PROT_DB_NAME)*
	rm -rf $(PROT_DB_PADDED)*
	rm -rf $(QUERY_DB_DIR)
	rm -rf $(RESULTS_DB_DIR)
	rm -rf $(TMP_DIR)
	rm -f $(M8_OUTPUT)