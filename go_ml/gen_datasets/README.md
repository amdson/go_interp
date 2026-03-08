# Evaluation Datasets

This directory contains notebooks for constructing the evaluation datasets used
in `dataset_eval/eval_models.ipynb`, plus a separate notebook (`dataset_gobench.ipynb`)
that generates the training data.

## Column Schema

All evaluation CSVs share this schema:

| Column | Type | Description |
|--------|------|-------------|
| `UniprotID` | str | UniProt accession (primary key) |
| `Sequence` | str | Amino acid sequence (≤ 850 residues) |
| `GOTerm` | str or list | Associated GO term(s) used as the functional label |
| `AnnotatedIndices` | list[int] | 0-based residue positions of functional annotations |

Some datasets include additional columns (e.g. `EnzymeClass`, `LigandID`, `InterproAccession`).

---

## Evaluation Datasets

### CSA — Catalytic Site Atlas (`csa_dataset.csv`)
**Source:** M-CSA (Mechanism and Catalytic Site Atlas)
- Data: https://www.ebi.ac.uk/thornton-srv/m-csa/media/flat_files/curated_data.csv
- EC→GO mapping: http://www.geneontology.org/external2go/ec2go

**Construction:** Extract catalytic residue indices from M-CSA entries; map EC numbers
to GO terms; filter sequences to ≤ 850 AA.

**Notebook:** `dataset_csa.ipynb`

---

### LLPS — Liquid-Liquid Phase Separation (`llps_dataset.csv`)
**Source:** PhasePro database
- Data: https://phasepro.elte.hu/download_full.json

**Construction:** Parse boundary coordinates of phase-separating regions; map organelle
annotations to GO terms; filter sequences ≤ 850 AA where the annotated region
covers < 75% of the sequence.

**Notebook:** `dataset_llps.ipynb`

---

### ELMs — Eukaryotic Linear Motifs (`elms_dataset.csv`)
**Source:** ELM database
- Instances: http://elm.eu.org/instances.tsv
- GO mappings: http://elm.eu.org/goterms.tsv
- Sequences (FASTA): http://elm.eu.org/instances.fasta

**Construction:** Download ELM instances and GO mappings; aggregate annotated indices
and GO terms per UniprotID; fetch sequences from UniProt; filter ≤ 850 AA.

**Notebook:** `dataset_elms.ipynb`

---

### BioLiP — Binding Sites (`biolip_dataset.csv`)
**Source:** BioLiP (Zhang Group, U Michigan)
- Data: https://zhanggroup.org/BioLiP/download/BioLiP_nr.txt.gz
- Sequences: https://zhanggroup.org/BioLiP/data/protein_nr.fasta.gz

**Construction:** Select top 9 ligand types (ADP, CA, CLA, MG, MN, ZN, DNA, peptide, RNA);
sample up to 120 entries per ligand; filter ≤ 850 AA; extract binding residue indices.

**Notebook:** `dataset_biolip.ipynb`

---

### InterPro Datasets (4 CSVs from one notebook)
**Source:** InterPro
- InterPro XML: https://ftp.ebi.ac.uk/pub/databases/interpro/releases/current/interpro.xml.gz
- InterPro→GO: https://ftp.ebi.ac.uk/pub/databases/interpro/releases/current/interpro2go
- Protein annotations: https://ftp.ebi.ac.uk/pub/databases/interpro/releases/current/protein2ipr.dat.gz
- Sequences: UniProt Swiss-Prot FASTA

**Construction:** Parse InterPro XML; filter to entries with > 10,000 annotated proteins;
sample up to 40 proteins per entry; filter ≤ 850 AA and domain coverage < 50% of sequence.
Produces separate CSVs for each annotation type:

| CSV | Annotation type |
|-----|----------------|
| `ip_domain_dataset.csv` | Protein domains |
| `ip_repeat_dataset.csv` | Repeat regions |
| `ip_active_site_dataset.csv` | Active sites |
| `ip_binding_site_dataset.csv` | Binding sites |

**Notebook:** `dataset_domain.ipynb`

> `ip_family_dataset.csv` and `ip_homologous_superfamily_dataset.csv` are also produced
> but were not included in the final evaluation.

---

## Training Data (not evaluation datasets)

### `dataset_gobench.ipynb` — CAFA5 training split

Generates the pre-split training and validation datasets used to train the
function-conditioned models. This notebook reads from the CAFA5 annotation data
and produces PyTorch pickle files.

**Outputs:**
- `../../data/train_esm_datasets/train_dataset.pkl`
- `../../data/train_esm_datasets/val_dataset.pkl`

> **Note:** This notebook has a hardcoded dependency on the CAFA5 dataset at
> `/home/andrew/cafa5_team/data/`. The key file `go_terms.json` has been copied
> to `../../data/go_terms.json`. The full CAFA5 annotation data must be obtained
> separately if re-running from scratch.

---

## Dropped Experiments

- `dataset_bidomain.ipynb` / `bidomain_dataset.csv` — multi-domain protein experiment, not in paper
- `dataset_pisite.ipynb` — protein-protein interface dataset, incomplete
