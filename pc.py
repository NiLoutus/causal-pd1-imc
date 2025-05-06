import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz
from sklearn.preprocessing import StandardScaler


# Function to run PC on a subset (IE1 or IE2)
def run_pc_for_domain(domain_label):
    df_sub = df_T[df_T['IE'] == domain_label].copy()
    df_sub = df_sub[variables].dropna()

    # Normalize features (PC assumes standardized continuous vars)
    X = StandardScaler().fit_transform(df_sub.values)

    # Run PC algorithm
    cg = pc(X, alpha=0.01, indep_test_method=fisherz)

    # Print edges with readable labels
    '''print(f"\n--- PC Graph for Domain {domain_label} ---")
    for edge in cg.G.get_graph_edges():
        i, j = edge[0], edge[1]
        print(f"{variables[i]} {edge[2]} {variables[j]}")

    # Optionally plot (requires Graphviz)
    GraphUtils.plot_graph(cg.G, labels=variables)'''
    return cg

# List of variables to include in PC
features = [col for col in df_T.columns if col.startswith('neigh_')]
variables = features + ['LAG-3', 'PD1']

# Encode column names to indices for causal-learn
var_idx_map = {var: i for i, var in enumerate(variables)}

# Run for both domains
gIE1 = run_pc_for_domain('IE1')
gIE2 = run_pc_for_domain('IE2')

gIE1.to_nx_graph()
G = gIE1.nx_graph

# Define custom labels (e.g., variable names in order of data matrix columns)
labels = {i: varname for i, varname in enumerate(variables)}

pos = nx.circular_layout(G)
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=1000, font_size=10, arrows=True)
plt.title("Causal Graph from PC IE1, k = 20")
plt.tight_layout()
plt.show()

gIE2.to_nx_graph()
G = gIE2.nx_graph

# Define custom labels (e.g., variable names in order of data matrix columns)
labels = {i: varname for i, varname in enumerate(variables)}

pos = nx.circular_layout(G)
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=1000, font_size=10, arrows=True)
plt.title("Causal Graph from PC IE2, k = 20")
plt.tight_layout()
plt.show()
