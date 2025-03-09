import numpy as np
import pandas as pd

df = pd.read_csv('idmapping_2025_03_08.tsv', sep='\t')
data = df.values  # Convert to a NumPy array
print(data)
k = 0
refseq_id_file = "refseq_test.txt"

file = open(refseq_id_file, 'w')
file2 = open("refseq_test2.txt", 'w')


for i,j in data:
    k = k + 1
    if k > 12500:
        file2.write(str(j) + "\n")
    else:
        file.write(str(j) + "\n")

