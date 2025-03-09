import numpy as np
import pandas as pd

df = pd.read_csv('idmapping_2025_03_08.tsv', sep='\t')
data = df.values  # Convert to a NumPy array
print(data)