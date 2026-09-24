# Quantum Score Matching

Code and data for the numerical simulations and IBM quantum experiments in [*Quantum score matching with applications to learning thermal states*](https://arxiv.org/abs/2609.28391), by Yulong Dong and Jiaqi Leng (2026).

## Contents

- [finite_shot](finite_shot/README.md): eight-qubit inhomogeneous TFIM learning with finite measurement budgets, 100 independent trajectories per configuration, population landscapes, and figure scripts.
- [ibm](ibm/README.md): four-qubit IBM experiments, native circuit construction, pre-experiment noise simulations, measurement outcomes, and classical parameter updates.

Each folder contains `Code`, `Data`, and `Figures`. Arrays use NumPy `.npz` files; settings and experimental metadata use JSON; tables use CSV. Figures can be regenerated from the included data without a cluster or an IBM account. Full simulations have separate dependencies and take longer.

## Reproduce the paper figures

Install NumPy, SciPy, Matplotlib and Pillow, then run from this directory:

```sh
python finite_shot/Code/make_figures.py
python ibm/Code/analyze.py
python ibm/Code/plot.py
```

## License and citation

Code and documentation are licensed under the [MIT License](LICENSE). Data and figures in the `Data` and `Figures` folders are licensed under [CC BY 4.0](LICENSE-DATA), subject to any separately stated third-party terms.

If you use this code or data in your research, please cite:

```bibtex
@article{dong2026quantumscorematching,
  title = {Quantum score matching with applications to learning thermal states},
  author = {Dong, Yulong and Leng, Jiaqi},
  journal = {arXiv preprint arXiv:2609.28391},
  year = {2026},
  url = {https://arxiv.org/abs/2609.28391}
}
```
