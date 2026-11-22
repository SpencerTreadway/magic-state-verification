# Magic State Verification

A numerical simulation of reconstructing, estimating, calibrating, and
certifying fidelity with respect to the single-qubit \(|T\rangle\) magic
state. The project is organized around four levels:

1. Bloch-vector tomography
2. \(X/Z\) fidelity estimation
3. Readout-error calibration
4. Sequential Hoeffding certification

The state is hidden behind `HiddenDevice` in the analysis notebooks: the
estimators receive only measurement outcomes and see the true state only for
post-hoc scoring.

## Requirements

- Python 3.12 or newer
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the notebooks

Start Jupyter:

```bash
jupyter lab
```

Then open `demo.ipynb` for the presentation walkthrough or `results.ipynb`
for the full Monte Carlo analysis.

In `results.ipynb`, set `QUICK = True` for a shorter noisy run or
`QUICK = False` for the larger final experiment. Figures are written to
`notebook_figures/`.

## Run the experiment script

The default script settings produce CSV tables and plots under
`experiment_results/`:

```bash
python experiments.py
```

## Repository layout

- `sim.py` - simulation, estimators, calibration, and certification primitives
- `notebook_utils.py` - test states, hidden device, statistics, and plotting helpers
- `experiments.py` - command-line Monte Carlo experiments
- `demo.ipynb` - live demonstration
- `results.ipynb` - complete numerical results and figures
- `report.md` - presentation-ready project report
- `experiment_results/` - CSV outputs and plots
- `notebook_figures/` - figures used by the notebooks
