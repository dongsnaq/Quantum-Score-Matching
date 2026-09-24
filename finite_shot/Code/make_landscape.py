"""Compute the J1,h1 population-loss slice used in the finite-shot figures."""
import argparse
from pathlib import Path
import numpy as np
from qsm import make_problem, loss_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--qubits', type=int, default=8)
    parser.add_argument('--points', type=int, default=101)
    parser.add_argument('--j-range', type=float, nargs=2, default=[0.3, 1.3])
    parser.add_argument('--h-range', type=float, nargs=2, default=[0.7, 1.8])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    problem = make_problem(args.qubits, args.beta)
    x, y = np.linspace(*args.j_range, args.points), np.linspace(*args.h_range, args.points)
    gap = np.zeros((len(y), len(x)))
    for row, h in enumerate(y):
        for col, j in enumerate(x):
            theta = problem.theta_star.copy()
            theta[0], theta[args.qubits-1] = j, h
            gap[row, col] = float(loss_only(theta, problem.terms, problem.frames,
                problem.target_state, problem.beta)) - problem.target_loss
        print(f'row {row+1}/{len(y)}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y, loss_gap=gap, theta_star=problem.theta_star,
        coupling_index=0, field_index=args.qubits-1)


if __name__ == '__main__':
    main()
