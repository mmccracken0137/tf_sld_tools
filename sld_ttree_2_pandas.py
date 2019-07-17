#!/usr/bin/env python
from ROOT import TTree, TFile, TLorentzVector
import sys
from root_numpy import root2array, rec2array
import numpy as np
import pandas as pd

# open root file and read branch names out of ttree...
tfile = TFile(sys.argv[1],'READ')
tree = tfile.Get('sldmuon_flat_Tree')

br_names = []
for b in tree.GetListOfBranches():
    br_names.append(b.GetName())

print('\n')

# remove features that should not be fit (e.g. truth values)
# from list of branches...
br_remove = ['weight', 'numtruepid_final',
             'truepids_decay', 'is_truetop', 'is_truecombo', 'is_bdtcombo',
             'beam_isgen', 'beam_beamid']
for b in br_names:
    if 'p4_meas' in b or 'p4_kin' in b:
        br_remove.append(b)
    elif 'x4_meas' in b or 'x4_kin' in b:
        br_remove.append(b)
    elif 'true' in b:
        br_remove.append(b)
    elif 'trkid' in b:
        br_remove.append(b)
    elif 'thrown' in b:
        br_remove.append(b)
print('ignoring the following features/branches in the tree:')
print(br_remove)
print('\n')

br_feats = [x for x in br_names if x not in br_remove]

print('ttree contains ' + str(len(br_names)) + ' features/branches')
print('ignoring ' + str(len(br_remove)) + ' features')
print('features file will contain ' + str(len(br_feats)) + ' features')
print('\n')

# root_numpy read ttree to numpy array
arr = root2array(sys.argv[1], "sldmuon_flat_Tree", br_feats)

# get types of features
br_types = {}
for i,v in enumerate(arr[0]):
    if type(v) == np.bool_:
        br_types[br_feats[i]] = np.int
    else:
        br_types[br_feats[i]] = type(v)
#print(br_types)

arr = rec2array(arr)

df = pd.DataFrame(data=arr, columns=br_feats)
df = df.astype(br_types)

print(df.head())
df.to_csv(sys.argv[1].split('.')[0].split('/')[-1] + '.csv')
