import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import fisherz
from sklearn.preprocessing import StandardScaler
import os
import sys


df_T = df_final[df_final['celltype'] == 'T_NK']

# List of variables to include in PC
features = [col for col in df_T.columns if col.startswith('neigh_')]
variables = features + ['LAG-3', 'PD1']

# Encode column names to indices for causal-learn
var_idx_map = {var: i for i, var in enumerate(variables)}

features = [col for col in df_T.columns if col.startswith('neigh_')] + ['LAG-3', 'PD1']

# Extract the data
data = df_T[features].copy()

# Encode the domain indicator
data['IE_code'] = df_T['IE'].astype('category').cat.codes

# Standardize the features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Combine features and domain indicator
data_matrix = data[features].values

# Set the index of the domain indicator (last column)
c_indx = data['IE_code'].values.reshape(-1, 1)

# Run CD-NOD
cg = cdnod(data_matrix, c_indx, alpha=0.01, indep_test_method=fisherz)

cg.to_nx_graph()
cg.draw_nx_graph(skel=False)

G = cg.nx_graph

# Define custom labels (e.g., variable names in order of data matrix columns)
labels = {i: varname for i, varname in enumerate(variables + ['IE'])}

# Optional: layout
pos = nx.circular_layout(G)

# Draw nodes and edges
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=1000, font_size=10, arrows=True)
plt.title("Causal Graph from CD-NOD, k = 20")
plt.tight_layout()
plt.show()
