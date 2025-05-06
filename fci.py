import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from sklearn.preprocessing import StandardScaler

# List of variables to include in PC
features = [col for col in df_T.columns if col.startswith('neigh_')]
variables = features + ['LAG-3', 'PD1']

# Encode column names to indices for causal-learn
var_idx_map = {var: i for i, var in enumerate(variables)}

def run_fci_for_domain(domain_label):
    df_sub = df_T[df_T['IE'] == domain_label][variables].dropna()

    # Standardize features for CI testing
    X = StandardScaler().fit_transform(df_sub.values)

    # Run FCI
    cg = fci(X, alpha=0.01, indep_test_method=fisherz)

    print(f"\n--- FCI Graph for Domain {domain_label} ---")


    return cg, variables

# Run for IE1 and IE2
cg_ie1, vars_ie1 = run_fci_for_domain('IE1')
cg_ie2, vars_ie2 = run_fci_for_domain('IE2')

# Map values to labels based on the rules
def interpret_edge(a_to_b, b_to_a):
    if a_to_b == 1 and b_to_a == 1:
        return "latent"
    elif a_to_b == 2 and b_to_a == 2:
        return "no_sep"
    elif a_to_b == 2 and b_to_a == 1:
        return "not_ancestor"
    elif a_to_b == -1 and b_to_a == 1:
        return "cause"
    else:
        return ""

fci_matrix = cg_ie1[0].graph

labels = []
for i in range(fci_matrix.shape[0]):
    row = []
    for j in range(fci_matrix.shape[1]):
        row.append(interpret_edge(fci_matrix[i][j], fci_matrix[j][i]))
    labels.append(row)

df_labels = pd.DataFrame(labels)

# Category color mapping
category_palette = {
    "": "white",
    "cause": "#1f77b4",
    "not_ancestor": "#ff7f0e",
    "no_sep": "#2ca02c",
    "latent": "#d62728"
}
category_to_num = {k: i for i, k in enumerate(category_palette.keys())}
df_numeric = df_labels.replace(category_to_num)
variable_names = [
    "neigh_B_cell", "neigh_T_NK", "neigh_aDC", "neigh_endothelial",
    "neigh_fibroblast", "neigh_myeloid", "neigh_neutrophil", "neigh_pDC",
    "neigh_plasma_cell", "neigh_stromal_undefined", "neigh_tumor", "LAG-3", "PD1"
]
df_numeric.index = variable_names
df_numeric.columns = variable_names
df_numeric_flipped = df_numeric.iloc[:, ::-1]


lut = [category_palette[k] for k in category_to_num.keys()]

# Plot
plt.figure(figsize=(10, 9))
sns.heatmap(df_numeric_flipped, cmap=lut, cbar=False, linewidths=0.5, linecolor='gray', square=True)
plt.title("FCI PAG Matrix IE1, k = 20")
plt.xlabel("Target Node")
plt.ylabel("Source Node")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()