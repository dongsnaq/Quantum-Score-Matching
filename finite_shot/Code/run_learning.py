from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from qsm import make_problem, population_point, population_points, loss_only  # noqa: E402
from resources import tail_bias_bound, truncated_resource_data, uniform_cutoff  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-shot QSM learning for an inhomogeneous TFIM."
    )
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--initialization", choices=("far", "local"), default="far")
    parser.add_argument("--initial-radius", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--local-learning-rate", type=float, default=0.5)
    parser.add_argument("--burn-in", type=int, default=60)
    parser.add_argument("--decay-time", type=float, default=10.0)
    parser.add_argument("--averaging-start", type=int)
    parser.add_argument("--maximum-relative-step", type=float, default=0.05)
    parser.add_argument("--tail-bias", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=203000)
    parser.add_argument("--output", type=Path, default=Path("learning_result.json"))
    return parser.parse_args()


def initialize(
    theta_star: np.ndarray,
    kind: str,
    radius: float,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    if kind == "far":
        return np.asarray([0.5] * (n - 1) + [1.0] * n)
    direction = rng.normal(size=theta_star.size)
    direction /= np.linalg.norm(direction)
    return theta_star + radius * np.linalg.norm(theta_star) * direction


def local_lcu_data(theta: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    coupling = theta[: n - 1]
    transverse = theta[n - 1 :]
    b = np.zeros(2 * n)
    q = np.zeros((2 * n, 2 * n - 1))
    for site in range(n):
        x_frame = 2 * site
        z_frame = x_frame + 1
        incident_bonds = [bond for bond in (site - 1, site) if 0 <= bond < n - 1]
        b[x_frame] = 2.0 * sum(abs(coupling[bond]) for bond in incident_bonds)
        b[z_frame] = 2.0 * abs(transverse[site])
        for bond in incident_bonds:
            q[x_frame, bond] = 2.0
        q[z_frame, n - 1 + site] = 2.0
    return b, q


def allocate_shots(total: int, ranges: np.ndarray) -> np.ndarray:
    if total < ranges.size:
        raise ValueError("the shot budget must be at least the number of parameters")
    weights = np.maximum(ranges, np.finfo(float).tiny)
    fractional = (total - ranges.size) * weights / weights.sum()
    extra = np.floor(fractional).astype(int)
    allocation = np.ones(ranges.size, dtype=int) + extra
    leftover = total - int(allocation.sum())
    if leftover:
        order = np.argsort(-(fractional - extra))
        allocation[order[:leftover]] += 1
    return allocation


def estimator_ranges(
    theta: np.ndarray,
    beta: float,
    cutoff: float,
    n: int,
) -> tuple[np.ndarray, float]:
    b, q = local_lcu_data(theta, n)
    ranges, _ = truncated_resource_data(beta, cutoff, b, q)
    bias = float(np.max(tail_bias_bound(beta, cutoff / beta, b, q)))
    return ranges, bias


def sample_gradient(
    exact_gradient: np.ndarray,
    ranges: np.ndarray,
    shots: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if not np.all(np.isfinite(exact_gradient)):
        raise ValueError("nonfinite population gradient")
    if np.any(np.abs(exact_gradient) > ranges * (1.0 + 1e-9)):
        raise ValueError("coefficient mass does not bound the gradient")
    probabilities = np.clip(0.5 * (1.0 + exact_gradient / ranges), 0.0, 1.0)
    plus_counts = rng.binomial(shots, probabilities)
    return ranges * (2.0 * plus_counts / shots - 1.0)


def learning_rate(args: argparse.Namespace, iteration: int) -> float:
    if args.initialization == "far" and iteration < args.burn_in:
        return args.learning_rate
    elapsed = iteration - args.burn_in if args.initialization == "far" else iteration
    return args.local_learning_rate / (1.0 + elapsed / args.decay_time)


def finite_difference_gradient(problem, theta: np.ndarray) -> np.ndarray:
    """Original fourth-order fallback for nonfinite eigenvector derivatives."""
    def value(point):
        return float(loss_only(point, problem.terms, problem.frames,
                               problem.target_state, problem.beta))
    gradient = np.empty_like(theta)
    for j in range(theta.size):
        h = 1e-4 * max(1.0, abs(float(theta[j])))
        offset = np.zeros_like(theta)
        offset[j] = h
        gradient[j] = (-value(theta+2*offset) + 8*value(theta+offset)
                       -8*value(theta-offset) + value(theta-2*offset))/(12*h)
    return gradient


def update(
    theta: np.ndarray,
    gradient: np.ndarray,
    beta: float,
    gram_diagonal: np.ndarray,
    rate: float,
    theta_star: np.ndarray,
    maximum_relative_step: float,
) -> np.ndarray:
    step = rate * beta ** -2 * gradient / gram_diagonal
    limit = maximum_relative_step * np.linalg.norm(theta_star)
    if np.linalg.norm(step) > limit:
        step *= limit / np.linalg.norm(step)
    return np.clip(theta - step, 0.0, 2.0)


def relative_error(theta: np.ndarray, theta_star: np.ndarray) -> float:
    return float(np.linalg.norm(theta - theta_star) / np.linalg.norm(theta_star))


def main() -> None:
    args = parse_args()
    if args.iterations is None:
        args.iterations = 300 if args.initialization == "far" else 200
    if args.averaging_start is None:
        args.averaging_start = args.iterations // 2
    if args.learning_rate is None:
        paper_rates = {0.2: 0.25, 0.4: 0.5, 0.6: 1., 0.8: 2.,
                       1.0: 4., 1.2: 8., 1.4: 16., 1.6: 32.}
        if args.initialization == "far" and args.beta not in paper_rates:
            raise ValueError("Specify --learning-rate for a temperature outside the paper grid")
        args.learning_rate = paper_rates.get(args.beta, 0.5)
    if not 0 <= args.averaging_start <= args.iterations:
        raise ValueError("averaging start must lie within the trajectory")
    if min(args.iterations, args.trajectories, args.batch_size) < 1 or args.beta <= 0:
        raise ValueError("iterations, trajectories, batch size and beta must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    problem = make_problem(args.qubits, args.beta, "local_tfim")
    _, target_gradient = population_point(problem, problem.theta_star)
    if not np.all(np.isfinite(target_gradient)) or np.linalg.norm(target_gradient) / np.sqrt(target_gradient.size) > 1e-10:
        raise RuntimeError("the target is not stationary at numerical precision")

    worst_theta = np.full(problem.theta_star.shape, 2.0)
    worst_b, worst_q = local_lcu_data(worst_theta, problem.n)
    cutoff = uniform_cutoff(args.beta, args.tail_bias, worst_b, worst_q)

    generators = [np.random.default_rng(args.seed + index) for index in range(args.trajectories)]
    theta = np.asarray(
        [
            initialize(problem.theta_star, args.initialization, args.initial_radius, rng, problem.n)
            for rng in generators
        ]
    )
    histories = [[] for _ in generators]
    theta_histories = [[point.copy()] for point in theta]

    for iteration in range(args.iterations):
        exact_gradients = np.concatenate([
            population_points(problem, theta[start:start + args.batch_size])[1]
            for start in range(0, len(theta), args.batch_size)
        ])
        used_fallback = ~np.all(np.isfinite(exact_gradients), axis=1)
        for run in np.flatnonzero(used_fallback):
            exact_gradients[run] = finite_difference_gradient(problem, theta[run])
        rate = learning_rate(args, iteration)
        for run, rng in enumerate(generators):
            ranges, certified_bias = estimator_ranges(theta[run], args.beta, cutoff, problem.n)
            shots = allocate_shots(args.shots, ranges)
            estimated_gradient = sample_gradient(exact_gradients[run], ranges, shots, rng)
            exact_norm = np.linalg.norm(exact_gradients[run])
            estimated_norm = np.linalg.norm(estimated_gradient)
            histories[run].append(
                {
                    "iteration": iteration,
                    "relative_parameter_error": relative_error(theta[run], problem.theta_star),
                    "absolute_gradient_error": float(
                        np.linalg.norm(estimated_gradient - exact_gradients[run])
                    ),
                    "exact_gradient_norm": float(exact_norm),
                    "gradient_alignment": float(
                        np.dot(estimated_gradient, exact_gradients[run])
                        / (estimated_norm * exact_norm)
                    )
                    if estimated_norm > 0 and exact_norm > 0
                    else 0.0,
                    "certified_tail_bias": certified_bias,
                    "used_gradient_fallback": bool(used_fallback[run]),
                    "coordinate_shots": shots.tolist(),
                    "coefficient_mass": ranges.tolist(),
                    "exact_gradient": exact_gradients[run].tolist(),
                    "estimated_gradient": estimated_gradient.tolist(),
                }
            )
            theta[run] = update(
                theta[run],
                estimated_gradient,
                args.beta,
                problem.gram_diagonal,
                rate,
                problem.theta_star,
                args.maximum_relative_step,
            )
            theta_histories[run].append(theta[run].copy())

    estimates = np.asarray(
        [np.mean(history[args.averaging_start :], axis=0) for history in theta_histories]
    )
    payload = {
        "model": {
            "name": "inhomogeneous TFIM",
            "qubits": problem.n,
            "beta": problem.beta,
            "theta_star": problem.theta_star.tolist(),
        },
        "optimization": {
            "shots_per_iteration": args.shots,
            "iterations": args.iterations,
            "trajectories": args.trajectories,
            "initialization": args.initialization,
            "learning_rate": args.learning_rate,
            "initial_radius": args.initial_radius,
            "seed": args.seed,
            "tail_bias": args.tail_bias,
            "parameter_bounds": [0.0, 2.0],
            "maximum_relative_step": args.maximum_relative_step,
            "local_learning_rate": args.local_learning_rate,
            "burn_in": args.burn_in,
            "decay_time": args.decay_time,
            "averaging_start": args.averaging_start,
            "maximum_evolution_time": float(2.0 * cutoff),
        },
        "trajectories": [
            {
                "seed": args.seed + run,
                "theta": np.asarray(theta_histories[run]).tolist(),
                "parameter_estimate": estimates[run].tolist(),
                "relative_parameter_error": relative_error(estimates[run], problem.theta_star),
                "history": histories[run],
            }
            for run in range(args.trajectories)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    errors = np.asarray([run["relative_parameter_error"] for run in payload["trajectories"]])
    print(f"wrote {args.output}")
    print(f"final mean relative error: {errors.mean():.6g}")
    if errors.size > 1:
        print(f"sample standard deviation: {errors.std(ddof=1):.6g}")


if __name__ == "__main__":
    main()
