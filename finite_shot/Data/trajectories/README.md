# Detailed trajectories

The eight `.npz` files here contain all 100 far-initialization runs at $10^5$ shots per iteration. Each companion JSON file gives the optimization settings.

- `theta`: `(run, iteration, parameter)`, shape `(100,301,15)`, including initialization.
- `theta_star`: the target parameter vector.
- `run`: run indices in the same order as `learning_summary.npz`.
- `seeds`: the original random seed for each run, in that same order.
- `parameter_estimate`: the final averaged parameter vector for each run, shape `(100,15)`.
- `endpoint_error`: relative error of each final averaged vector.
- `learning_error`, `loss_gap`, `gradient_error`, `gradient_norm`, `gradient_alignment`, `learning_rate`: `(run, iteration)`, shape `(100,300)`, evaluated before each update.

All 96 configurations have their 100 individual endpoint errors in `../learning_summary.npz`. That file also stores learning-error histories at $10^6$ shots. The `../trajectory_slices` files preserve all 100 projected paths needed for the landscape figure at $10^3$ and $10^5$ shots. Full parameter histories for other configurations are not part of this compact figure dataset.
