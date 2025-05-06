import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
import sys

counts = pd.read_csv('./data/Protein_panel_singlecell_counts.txt', sep='\t', index_col = 0)
metadata = pd.read_csv('./data/Protein_panel_singlecell_metadata.csv', sep=',', index_col = 0)
sel_counts = counts.T[['LAG-3', 'PD1']]
sel_metadata = metadata[['Center_X', 'Center_Y', 'celltype', 'sample', 'ROI', 'IE']]
df = pd.concat([sel_metadata, sel_counts], axis=1)
df_celltype = pd.get_dummies(df['celltype'])
df = pd.concat([df, df_celltype], axis=1)

# Create a placeholder for storing neighborhood compositions
composition_results = []

# Group by sample and ROI
grouped = df.groupby(['sample', 'ROI'])

for (sample, roi), group in grouped:
    coords = group[['Center_X', 'Center_Y']].values
    celltypes = group[df_celltype.columns].values  # one-hot matrix
    idx = group.index.to_numpy()

    # Use Nearest Neighbors (exclude the point itself)
    nn = NearestNeighbors(n_neighbors=11, algorithm='ball_tree')  # include self
    nn.fit(coords)
    neighbors = nn.kneighbors(coords, return_distance=False)

    # Compute composition for each cell (excluding itself)
    for i, indices in enumerate(neighbors):
        neighbor_indices = indices[indices != i][:20]  # exclude self and limit to 20
        neighbor_celltypes = celltypes[neighbor_indices]
        composition = neighbor_celltypes.mean(axis=0)  # frequency per cell type
        result = {
            'index': idx[i],
            **{f'neigh_{ct}': composition[j] for j, ct in enumerate(df_celltype.columns)}
        }
        composition_results.append(result)

df_composition = pd.DataFrame(composition_results).set_index('index')
df_final = df.join(df_composition)
df_final.to_csv('./data/Protein_panel_singlecell_counts_composition_10.csv', sep=',')

# df_final = pd.read_csv('./data/Protein_panel_singlecell_counts_composition_10.csv', sep=',', index_col=0)

df_T = df_final[df_final['celltype'] == 'T_NK']