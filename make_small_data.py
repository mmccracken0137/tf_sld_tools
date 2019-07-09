import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys

file = sys.argv[1]

# read in feats files
print('\nsmallening data file...\n')
df_sld = pd.read_csv(file)

x_train, x_test = train_test_split(df_sld, test_size=0.02, random_state=42)

outfile = sys.argv[1].split('/')[-1].split(".")[0] + '_small.csv'
print('writing ' + outfile + '\n')
x_test.to_csv(outfile, index=False)
