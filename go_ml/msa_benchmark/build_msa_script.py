import subprocess, os

def run_muscle(input_fasta, output_aligned):
    muscle_args = ["/home/andrew/GO_interp/muscle", "-super5", input_fasta, "-output", output_aligned]
    subprocess.run(muscle_args, check=True)

def run_muscle_dir(fasta_dir = "uniref_msa/csa_msa/", msa_dir = "uniref_msa/csa_msa_output/"):
    from Bio import SeqIO
    for fasta_file in os.listdir(fasta_dir):
        fasta_path = os.path.join(fasta_dir, fasta_file)
        # sequences = list(SeqIO.parse(fasta_path, "fasta"))
        # max_len = max(len(seq.seq) for seq in sequences)
        # if max_len > 850:
        #     continue
        if fasta_file.endswith(".fasta"):
            fasta_path = os.path.join(fasta_dir, fasta_file)
            output_aligned = os.path.join(msa_dir, f"{os.path.splitext(fasta_file)[0]}_aligned.fasta")
            if not os.path.exists(msa_dir):
                os.makedirs(msa_dir)
            if os.path.exists(output_aligned):
                print(f"Skipping {fasta_file}, already aligned.")
                continue
            run_muscle(fasta_path, output_aligned)
            print(f"Alignment done for {fasta_file}")
    print("Finished obtaining MSAs")

# run_muscle_dir('uniref_msa/csa_msa/', 'uniref_msa/csa_msa_output/')
# run_muscle_dir('uniref_msa/elms_msa/', 'uniref_msa/elms_msa_output/')
# run_muscle_dir('uniref_msa/llps_msa/', 'uniref_msa/llps_msa_output/')
run_muscle_dir('uniref_msa/ip_domain_msa/', 'uniref_msa/ip_domain_msa_output/')
