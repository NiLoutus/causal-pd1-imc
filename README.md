# causal-pd1-imc
This repository contains the code and analysis pipeline for identifying causal regulators of PD-1 expression in spatial immune microenvironments using IMC data and causal discovery methods including PC, FCI, and CD-NOD.

## Project Structure

- `preprocessing.py`: Prepares input data, computes neighborhood features using k-nearest neighbors, and encodes domain labels (IE1 vs IE2).
- `pc.py`: Applies the PC algorithm separately within each immune domain to identify candidate causal links under causal sufficiency.
- `fci.py`: Runs the FCI algorithm to account for latent confounders and outputs partially oriented causal graphs (PAGs).
- `cdnod.py`: Uses CD-NOD with domain labels to discover stable, invariant causal relationships across immune contexts.

## Requirements

Install required dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn networkx causallearn
