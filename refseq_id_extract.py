import numpy as np
import pandas as pd

df = pd.read_csv('data/idmapping_2025_03_08.tsv', sep='\t')
data = df.values  # Convert to a NumPy array
print(data)
k = 0

refseq_id_file = "refseq_test.txt"

file0 = open("refseq_test0.txt", 'w')
file1 = open("refseq_test1.txt", 'w')
file2 = open("refseq_test2.txt", 'w')
file3 = open("refseq_test3.txt", 'w')


for i,j in data:

    k = k + 1
    if k < 7000:
        file0.write(str(j) + "\n")
    elif k >= 21000:
        file3.write(str(j) + "\n")
    elif k >= 14000:
        file2.write(str(j) + "\n")
    elif k >= 7000:
        file1.write(str(j) + "\n")




