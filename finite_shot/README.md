# Finite-shot inhomogeneous TFIM learning

The model is the open-chain Hamiltonian

$$H(\theta)=-\sum_{i=1}^{n-1}J_i Z_iZ_{i+1}-\sum_{i=1}^{n}h_i^x X_i.$$

All $2n-1$ coefficients are learned independently using the frame $\{X_i,Z_i\}_{i=1}^{n}$. The supplied data use $n=8$, target couplings $J_i^*=1$, and target fields $(h_i^x)^*=1.5$. There are 100 independent trajectories at each of eight temperatures, six shot budgets, and two initializations.

## Figures

From the repository root:

```sh
python finite_shot/Code/make_figures.py
python finite_shot/Code/make_animation.py --steps 300 --stride 6
```

These commands read the saved data and require NumPy, Matplotlib and Pillow. Outputs go to `Figures`. Error bands and endpoint ellipses show one sample standard deviation. Gradient diagnostics use the recorded points without smoothing. The landscape is a population-loss slice in $J_1,h_1^x$ with the other parameters fixed at their target values; the displayed paths are projections of full 15-parameter trajectories.

## New simulations

Install `finite_shot/requirements.txt` in a separate Python environment. A short check is:

```sh
python finite_shot/Code/run_learning.py --qubits 4 --beta 0.2 --shots 1000 --iterations 4 --trajectories 2 --output example.json
```

For a paper-sized configuration:

```sh
python finite_shot/Code/run_learning.py --qubits 8 --beta 0.6 --shots 100000 --initialization far --trajectories 100 --seed 203000 --output learning_beta06.json
```

The code uses full-system exact diagonalization. It evaluates the population objective and gradient in `qsm.py`, computes coefficient masses in `resources.py`, then draws the finite-shot gradient in `run_learning.py`. Each coordinate has the conditional distribution

$$B_j\sim\operatorname{Binomial}(N_j,(1+g_j/L_j)/2),\qquad \widehat g_j=L_j(2B_j/N_j-1).$$

Here $g_j$ is the exact population gradient. As in Appendix E.1, the simulation centers the estimator at $g_j$ to isolate measurement fluctuations; the finite time cutoff enters the coefficient mass only. It does not simulate circuit noise or time-discretization bias. The random-time and measurement draws are represented by their combined binary outcome law. Shots are allocated in proportion to $L_j$, with integer rounding preserving the total budget. No amplitude amplification is used.

## Optimization settings

The temperatures are $0.2,0.4,\ldots,1.6$ and total shots per iteration are $10^3,3\times10^3,10^4,3\times10^4,10^5,10^6$. Parameters are projected onto $[0,2]^{15}$. Updates use $\beta^{-2}\Gamma^{-1}$ with $\Gamma=\operatorname{diag}(8I_7,4I_8)$ and a step-norm cap of $0.05\|\theta^*\|_2$.

- Far initialization: all $J_i=0.5$, all $h_i^x=1$, 300 updates. The first 60 updates use rates $0.25,0.5,1,2,4,8,16,32$ in increasing temperature order; subsequent rates are $0.5/(1+(t-60)/10)$. The final estimate averages iterates 150 through 300, inclusive.
- Local initialization: an independent Gaussian direction at relative distance 0.05 from the target, 200 updates, rate $0.5/(1+t/10)$. The final estimate averages iterates 100 through 200, inclusive.

The script selects these defaults from temperature and initialization. Learning curves show errors of raw iterates; shot-scaling endpoints use the averaged parameter estimate. Different CPU/GPU eigensolvers can lead to different noisy trajectories; the archived data provide the original figure values.

Gradient evaluations use batches of one by default to limit memory use. `--batch-size` can be increased on a machine with sufficient memory. A new noiseless slice can be computed with `Code/make_landscape.py --beta 0.6 --output landscape.npz`; the saved paper grids also record their axes explicitly.

As in the original simulation, a nonfinite eigenvector derivative triggers a fourth-order finite-difference fallback with relative step $10^{-4}$. Its use is recorded for each update.

## Data

- `learning_summary.npz`: `endpoint_error` has axes `(initialization, beta, shots, run)` and shape `(2,8,6,100)`. `learning_error_far` and `learning_error_local` contain the raw-iterate errors at $10^6$ shots, with axes `(beta, run, iteration)`.
- `gradient_diagnostics.npz`: per-run gradient errors, exact-gradient norms, cosine alignments, and learning errors at $10^5$ shots for far initialization and $\beta=0.2,0.8,1.6$.
- `landscapes`: noiseless two-coordinate population-loss grids at $\beta=0.2,0.6,1.0,1.6$.
- `trajectory_slices`: 100 projected paths at $10^3$ and $10^5$ shots for the landscape panels.
- `trajectories`: per-run parameter histories, diagnostics and optimization settings. Each file corresponds to one temperature, initialization, and shot budget.

`run_learning.py` saves parameter histories, exact and sampled gradients, coordinate shot counts, coefficient masses and the final averaged estimate. The archived experiment records contain the measurements and diagnostics retained by the original runs; their available arrays are documented in `Data/trajectories/README.md`.
