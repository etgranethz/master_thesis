# Analysis of Thalamocortical Data During a Reach Task to Test Models of Neural Control of Action Sequences

## Project Overview

This repository contains code and analysis on the investigation of **how the brain controls sequences of actions** in a skilled behavior. The study focuses on a **mouse reach-to-grasp task** and uses simultaneous thalamic and cortical neural recordings (thalamocortical data).


By analyzing the neural activity alongside the kinematics of the reaching movement, the project tests whether the data show evidence of discrete **trigger events** (supporting a hierarchical, sequential organization of action). The code in this repository processes the behavioral data (e.g. paw trajectories, grip aperture) and neural data (from cortex) to identify patterns consistent with either model.

## Repository Structure

The repository is organized into the following components:

- **`data_loading.py`** – Functions to load and preprocess raw experimental data.
- **`behavioural_utils.py`** – Utilities for behavioral (kinematic) analysis, such as computing grip aperture and identifying key movement events.
- **`neural_utils.py`** – Methods for neural data analysis, including identification of neural trigger dimensions and cross-correlation with behavior.
- **`Data_analysis.ipynb`** – A Jupyter Notebook that reproduces the full analysis pipeline and generating figures used in the thesis.
- **`mit_requirements.txt`** – List of Python packages and versions required to reproduce the analysis.

## Installation

Create a virtual environment (recommended) and install dependencies:

```bash
pip install -r mit_requirements.txt
```

## Usage

Open `Data_analysis.ipynb` in Jupyter Notebook or JupyterLab. The notebook will:

1. **Load data** from experimental recordings.
2. **Perform behavioral analysis** (e.g., compute grip aperture, segment movement phases).
3. **Compute neural trigger signals** by training a linear classifier on neural data around movement phases.
4. **Generate figures** illustrating neural vs behavioral dynamics to test hierarchical vs continuous control models.

Ensure the raw data files are available in the expected paths or update the file paths at the top of the notebook as needed.

---

*This repository was developed Ettore Gran's master's thesis at the McGovern Institute, MIT (April 30, 2025).*
