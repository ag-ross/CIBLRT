# CIB-LRT: Beyond Consistent Scenarios: Deriving Indirect Influence, Transition Resistance, and Adjustment Dynamics

Code for applying Linear Response Theory (LRT) to Cross-Impact Balance (CIB) analysis. The framework extends standard CIB beyond attractor enumeration, deriving four analytical objects in closed form from the cross-impact matrix: a cross-impact multiplier, a susceptibility matrix, an impulse response function (IRF), and a unit-impulse shock profile.

## Repository layout

**`replicate.py`** — Single entry point that runs the full replication pipeline in one command.

**`workings/`** — Full replication of the paper pipeline for the 15-descriptor energy-transition dataset (`Phase_D_CIM.csv`): attractor enumeration, all LRT objects, figures, and sensitivity analyses. See `workings/README.md`.

**`examples/`** — Standalone practitioner examples with hardcoded CIMs, requiring no external data files. See `examples/README.md`.

## Quick start

To replicate all paper results (figures, tables, and CSV data) in one command:

```
python replicate.py          # run full pipeline; outputs -> workings/outputs/
python replicate.py --force  # recompute LRT objects from cached attractor set
```

Or run the pipeline stages individually from the `workings/` directory:

```
cd workings
python run_analysis.py          # compute and cache LRT objects (run first)
python run_analysis.py --force  # recompute, preserving the locked attractor set
python plot_irf.py              # figures FIG_01, FIG_03, FIG_04, FIG_S1-S3
python shock_descriptor.py      # unit-impulse shock figure FIG_02
python export_tables.py         # paper tables as CSV
```

To run the self-contained practitioner example:

```
cd examples
python example_policy_shock.py
```

## Dependencies

All results were compiled with Python 3.9.6.

```
pip install numpy scipy matplotlib
pip install git+https://github.com/ag-ross/PyCIB.git
```

scipy is required for the `workings/` pipeline. A minimal numpy-only fallback shim is provided in `workings/_lrt_shim/` if scipy is not installed. As a fallback for PyCIB [2], all `workings/` scripts will also accept a directory named `PyCIB-main/` placed at the project root instead of a pip installation.

## Data and code

`workings/Phase_D_CIM.csv` contains the cross-impact matrix from [5]. The network graph visualisation in Figure 4 uses plotting utilities from [5,6].

## Licence

Details are provided in the LICENSE file.

### Cite as:

Ross, A. G. & Gershenzon, J. & Kleefeld, A. (2026). Beyond Consistent Scenarios: Deriving Indirect Influence, Transition Resistance, and Adjustment Dynamics. 

## References

[1] Klimek, P., Poledna, S., & Thurner, S. (2019). Quantifying economic resilience from input–output susceptibility to improve predictions of economic growth and recovery. Nature communications, 10(1), 1677. https://doi.org/10.1038/s41467-019-09357-w

[2] Ross, A. G. (2026). PyCIB: Cross-Impact Balance (CIB) Analysis Package [Software]. https://github.com/ag-ross/PyCIB. https://doi.org/10.5281/zenodo.18367511

[3] Raseta, M., Kleefeld, A., Grajewski, M., & Ross, A. G. (2024). Green-Kubo Relations for Networked Economic Systems: An Exact Solution for Quantifying Economic Resilience from Input-Output Susceptibility. Available at SSRN 4936915. http://dx.doi.org/10.2139/ssrn.4936915 

[4] Ross, A. G., Raseta, M., Grajewski, M., & Kleefeld, A. (2025). Resilience and recovery in networked economic systems: An ex-ante analysis of susceptibility to aspects of a potential China–Taiwan conflict. Papers in Regional Science, 104(3), 100094. https://doi.org/10.1016/j.pirs.2025.100094

[5] Ross, A. G., & Ross, A. M. (2026). AI-Simulated Expert Panels for Socio-Technical Scenarios and Decision Guidance. arXiv preprint arXiv:2603.29470. https://doi.org/10.48550/arXiv.2603.29470

[6] Ross, A. G. (2026). From transient shocks to unexpected outcomes: disruptive drivers in scenario pathways. arXiv preprint arXiv:2604.20879. https://doi.org/10.48550/arXiv.2604.20879 

(Please check for the latest versions of 5 and 6 as a peer-reviewed version may be available.)
