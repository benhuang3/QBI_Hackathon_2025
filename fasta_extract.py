from Bio import SeqIO
import pandas as pd


uni_ref_dict = {}
ref_nuc_dict = {}

df = pd.read_csv('idmapping_2025_03_08.tsv', sep='\t')
data = df.values  # Convert to a NumPy array
for i,j in data:
    uni_ref_dict[i] = j



fasta_file = "sequence.fasta"
fasta_file2 = "sequence(1).fasta"

# Parse the FASTA file
for record in SeqIO.parse(fasta_file, "fasta"):
    ref_nuc_dict[record.id] = record.seq


for record in SeqIO.parse(fasta_file2, "fasta"):
    ref_nuc_dict[record.id] = record.seq

print(uni_ref_dict["P48347-2"] == "NM_179367.2")
#print(ref_nuc_dict['NM_001252541.1'])

#print(ref_nuc_dict["NM_179367.2"])

combined_dict = {key: value + str(ref_nuc_dict[value]) for key, value in uni_ref_dict.items() if value in ref_nuc_dict}
