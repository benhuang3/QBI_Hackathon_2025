from Bio import SeqIO
import gzip

# Path to the UniProt nucleotide FASTA file
fasta_file = "data/uniprot_sprot_varsplic.fasta"
id_file = "data/uniprot_ids.txt"

# Dictionary to store UniProt ID → Nucleotide Sequence mapping
uniprot_nucleotide_dict = {}

file = open(id_file, 'w')


# Open and parse the FASTA file
with open(fasta_file, "r") as f:
    for record in SeqIO.parse(f, "fasta"):
        uniprot_id = record.id.split("|")[1]  # Extract UniProt ID
        uniprot_nucleotide_dict[uniprot_id] = str(record.seq)
        file.write(str(uniprot_id) + "\n")


# Print first few entries
for k, v in list(uniprot_nucleotide_dict.items())[:5]:
    print(f"UniProt ID: {k}\nNucleotide Sequence: {v[:50]}...\n")