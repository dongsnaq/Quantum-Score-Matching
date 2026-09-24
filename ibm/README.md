# Quantum score matching on IBM hardware

Data and offline reproduction code for learning a four-qubit transverse-field Ising Hamiltonian on `ibm_pittsburgh`.

## Experiment

The open-chain Hamiltonian is

$$H(J,h)=-J\sum_{i=1}^{3}Z_iZ_{i+1}-h\sum_{i=1}^{4}X_i.$$

The target is $(J^*,h^*)=(1,1.5)$ and every trajectory starts at $(J_0,h_0)=(0.5,0.5)$. The derivation frame contains the eight single-qubit operators $X_i,Z_i$. Both homogeneous parameters are learned. The inverse temperatures are $\beta=0.2,0.4$, with five independently randomized trajectories at each temperature and 45 parameter updates per trajectory. Here a seed distinguishes the random measurement circuits and state-preparation choices, not the initialization.

After 45 updates, the mean relative parameter error is 10.06% at $\beta=0.2$ and 14.78% at $\beta=0.4$, with sample standard deviations of 5.72 and 6.91 percentage points across five trajectories. Separate coupling and field errors are in [Data/final_errors.csv](Data/final_errors.csv). The complete learning curves are in [Figures/hardware_learning_errors.pdf](Figures/hardware_learning_errors.pdf).

At each update, each trajectory uses 256 random circuit instances, allocated between the two gradient coordinates in proportion to their coefficient masses. Each instance is measured 32 times, giving 8192 measurements per trajectory per update. The random times, Pauli terms and target eigenstate are held fixed within these 32 repetitions. They are sampled again for the next instance.

Each local circuit uses four system qubits and one ancilla. Instances from all ten trajectories are randomly assigned to 19 disjoint five-qubit paths on the 156-qubit processor. The paths have minimum graph distance two. One epoch contains 2560 local instances packed into 135 processor circuits, each executed for 32 shots; the last processor circuit uses 14 paths. Across 45 epochs, the experiment records 3,686,400 ancilla bits in 6075 processor circuits. The reported total QPU usage is 180 seconds. Parallel block readout gives multiple local measurement outcomes in one processor shot.

The target Gibbs state is supplied by classically assisted preparation: a target energy eigenstate is drawn with its Boltzmann probability and prepared with native gates. This eigenstate is fixed for the 32 executions of its circuit instance, but the physical state is freshly prepared on each execution. The target Hamiltonian is used to prepare benchmark data, not to supply the gradient to the optimizer. Candidate evolution uses a second-order product formula with maximum step size 0.25. The native gate set is `rz`, `sx`, `x`, `cz`, and `measure`.

## Files

```text
Code/
  qsm_ibm_model.py       QSM objective, randomized terms and parameter updates
  circuits.py           native circuit construction
  simulate.py           pre-experiment noise simulation
  analyze.py            measurement analysis
  plot.py               hardware figures
Data/
  experiment.json       experimental and optimizer settings
  layout.json           physical qubit paths
  compiler.json         five-wire routing settings
  layout_calibration.json
  software.json         recorded software versions
  target_states.npz     target spectrum, eigenvectors and probabilities
  state_preparations.npz
  trajectories.csv      parameters, gradients and errors at every epoch
  trajectories.npz      the same trajectories as numerical arrays
  final_errors.csv      final mean errors and sample standard deviations
  circuit_resources.csv
  execution_summary.json
  validation.json
  circuit_validation.json
  population_landscape.npz
  epoch001/ ... epoch045/
    circuits.npz        actual submitted native gate lists as ordinary arrays
    instances.json     mapping from circuits/registers to random instances
    outcomes.npz       per-shot ancilla measurements
    counts.json        zero/one counts for each register
    optimization.json  parameter values before/after the update and gradients
    execution.json     job identifier, timestamps, options and result metadata
    calibration.json
Figures/                learning curves, trajectories and mean-path animation
```

Epoch directories are one-based: `epoch001` contains the first measurement and update. The `epoch` field inside the original numerical records is zero-based. In `trajectories.csv`, epoch zero is the initialization and epoch 45 is the final state. The ten groups are named, for example, `b0.2_seed0`.

The official IBM job identifiers in `execution.json` identify the hardware acquisitions. Circuit arrays retain every native instruction and rotation angle from the submitted circuits; the construction code additionally permits offline reconstruction. No IBM account is needed to read, validate or plot this dataset.

## Reproduce the results

### Pre-experiment circuit simulation

Install `requirements-simulation.txt` for the Aer simulator and the calibration reader. From the repository root:

```sh
python ibm/Code/simulate.py --beta 0.2 --circuits 256 --shots 8192 --steps 40 --seed 0 --workers 4 --output simulation_beta02_seed0.json
```

For a short execution check, set `--circuits 4 --shots 128 --steps 1`. Use `--ideal` to disable the noise model. Each instance prepares a sampled target eigenstate through native gates, implements the randomized gradient circuit, and measures the ancilla. The eigenstate and random times stay fixed for all repetitions of that instance. The optimizer uses the measured gradient. Every completed update saves the parameters, sampled instances, counts and random-generator states. Add `--resume` and increase `--steps` to continue that output file.

`Data/simulation` contains the ten pre-experiment trajectories for $\beta=0.2,0.4$, seeds 0--4, initialization $(0.5,0.5)$, 256 instances, 8192 shots and 40 updates. `calibration.json` is the saved calibration used for those simulations. These are five-qubit, single-block noise simulations on physical qubits 83--87; they do not model crosstalk between the 19 blocks used on hardware. The recorded hardware outcomes are in the epoch folders.

### Recorded hardware data

The recorded environment used Python 3.9. In a compatible Python environment, run from this directory:

```sh
python -m pip install -r requirements.txt
python Code/analyze.py
python Code/plot.py --gif
```

`analyze.py` starts from every recorded bitstring, checks its aggregate counts, reconstructs each gradient, and verifies every parameter update against the stored optimizer states. It writes the trajectory tables, final-error table and numerical validation report. Plotting uses the saved noiseless population landscape. To recompute that landscape by exact diagonalization, use:

```sh
python Code/plot.py --recompute-landscape
```

For a full check of circuit mappings, native operations, random sampling and representative reconstructed circuits:

```sh
python Code/validate_circuits.py
```

This checks every submitted circuit and every random instance, and reconstructs two local circuits in each epoch for an instruction-by-instruction comparison. Reconstruct all packed circuits of an individual epoch with:

```sh
python Code/circuits.py --epoch 1 --output reconstructed_epoch001.npz
```

All commands are offline and make no hardware submissions. Numerical analysis requires only NumPy; the circuit and landscape commands also use SciPy and Qiskit. `Data/software.json` records the original execution-related packages, including the runtime and simulator versions, which are not required for reading the hardware measurements.

## Measurement and optimization conventions

For coordinate $j$, instance $\ell$ has coefficient sign $s_\ell\in\{-1,1\}$. Its signed readout mean is

$$y_{j\ell}=s_\ell\frac{n_{0,j\ell}-n_{1,j\ell}}{32},\qquad \widehat g_j=L_j\frac{1}{K_j}\sum_{\ell=1}^{K_j}y_{j\ell}.$$

Here $K_j$ is the number of random instances allocated to that coordinate and $L_j$ is its recorded truncated coefficient mass. The mass includes the sum over all frame generators and all Hamiltonian terms contributing to the homogeneous derivative.

At zero-based iteration $t$, form

$$u_t=-\frac{0.5}{1+t/10}\,\frac{1}{\beta^2}\operatorname{diag}(24,16)^{-1}\widehat g_t.$$

Rescale $u_t$ if needed so that $\|u_t\|_2\leq0.05\|\theta^*\|_2$, then set $\theta_{t+1}=\Pi_{[0,2]^2}(\theta_t+u_t)$. The step cap uses the fixed target norm of this benchmark. These are raw parameter iterates, not running averages.

The time cutoff bounds the omitted integral tail by $10^{-4}$ uniformly over the candidate box. Random absolute times are sampled from tabulated inverse CDFs with 32768 positive grid points, either with or without the first-moment importance weight according to the response term. The sampled times are recorded in `instances.json`.

`instances.json` lists processor circuits under `pubs`, in submission order. For processor circuit index 0 and its first block, `outcomes.npz['pub000_block0']` is a uint8 array of shape `(32, 1)`. Its entries are the measured bits. The same register's counts appear in `counts.json[0]['block0']`. Instance records specify the parameter coordinate (`0=J`, `1=h`), frame index, coefficient sign, time variables, Pauli insertions, target eigenstate, routed ancilla position and physical path. Frame indices are zero-based in the order `X0,Z0,X1,Z1,X2,Z2,X3,Z3`; Pauli strings use Qiskit's rightmost-character-for-qubit-zero convention.

Raw shot ordering is preserved within each processor circuit, including across its block registers. It can therefore be used to study cross-block outcome correlations. Outcomes from different processor circuits should not be treated as simultaneous shots.

The array axes of `trajectories.npz` are `(temperature, seed, epoch, parameter)` for `theta`, with shape `(2,5,46,2)`. Gradient, mass and instance-count arrays have 45 rather than 46 epochs. Each gradient at array index $t$ takes `theta[...,t,:]` to `theta[...,t+1,:]`. The CSV provides the same information without requiring an array library.

### Reading a circuit without Qiskit

Each `circuits.npz` contains plain numeric and string arrays, with no Python objects or pickle. Gate arrays from the 135 processor circuits are concatenated; `circuit_offsets[i:i+2]` specifies the slice for circuit `i`. `gate` gives the operation name, `qubit0` and `qubit1` give physical qubits, `angle` gives the `rz` angle in radians, and `clbit` gives the destination of a measurement. A missing second qubit or classical bit is `-1`; the angle is zero for gates other than `rz`. Circuit-level arrays give `num_qubits`, `num_clbits` and `global_phase` in radians. Classical registers are one bit each, indexed by `register_offsets` into `register_names`.

```python
import numpy as np

with np.load('Data/epoch001/circuits.npz', allow_pickle=False) as circuits:
    start, stop = circuits['circuit_offsets'][:2]
    gates = circuits['gate'][start:stop]
    angles = circuits['angle'][start:stop]
```

`Code/circuit_data.py` provides `load_circuit(path, index)` to convert one stored gate list into a Qiskit circuit. `state_preparations.npz` uses the same format for the 16 five-qubit state-preparation circuits.

## Figures

- `hardware_learning_errors.pdf`: overall relative Euclidean error and separate relative $J,h$ errors. Curves show the mean of five trajectory errors; shaded bands are one sample standard deviation (`ddof=1`). No smoothing is used.
- `hardware_seed_trajectories.pdf`: all five measured trajectories at each temperature on the noiseless population loss difference $\mathcal J(\theta)-\mathcal J(\theta^*)$, computed by exact diagonalization. A floor of $10^{-7}$ is applied for the logarithmic display; the saved data retain the original loss values.
- `hardware_mean_dynamics.gif`: the pointwise mean of the five hardware trajectories through all 45 updates. The corresponding PDF is the final frame.
