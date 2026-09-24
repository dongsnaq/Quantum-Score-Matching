from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from models import pauli_string


jax.config.update("jax_enable_x64", True)

COMPLEX = jnp.complex128
REAL = jnp.float64


@dataclass(frozen=True)
class ExactQSMProblem:
    model_name: str
    n: int
    beta: float
    terms: jax.Array
    frames: jax.Array
    theta_star: np.ndarray
    target_state: jax.Array
    target_loss: float
    gram_diagonal: np.ndarray


def target_theta(n: int, model_name: str = "local_tfim") -> np.ndarray:
    if model_name != "local_tfim":
        raise ValueError("This example implements inhomogeneous TFIM learning")
    return np.asarray([1.0] * (n - 1) + [1.5] * n)


def build_dense_model(n: int, model_name: str = "local_tfim") -> tuple[np.ndarray, np.ndarray]:
    if model_name != "local_tfim":
        raise ValueError("This example implements inhomogeneous TFIM learning")
    bonds = [-pauli_string(n, {site: "Z", site + 1: "Z"}) for site in range(n - 1)]
    transverse_fields = [-pauli_string(n, {site: "X"}) for site in range(n)]
    terms = bonds + transverse_fields
    frames = [
        operator
        for site in range(n)
        for operator in (
            pauli_string(n, {site: "X"}),
            pauli_string(n, {site: "Z"}),
        )
    ]
    return np.asarray(terms), np.asarray(frames)


def gibbs_state(hamiltonian: jax.Array, beta: float) -> jax.Array:
    energies, eigenvectors = jnp.linalg.eigh(hamiltonian)
    weights = jnp.exp(-beta * (energies - energies[0]))
    probabilities = weights / jnp.sum(weights)
    return (eigenvectors * probabilities) @ eigenvectors.conj().T


def qsm_loss(
    theta: jax.Array,
    terms: jax.Array,
    frames: jax.Array,
    target_state: jax.Array,
    beta: float,
) -> jax.Array:
    hamiltonian = jnp.tensordot(theta, terms, axes=1)
    energies, eigenvectors = jnp.linalg.eigh(hamiltonian)
    adjoint = eigenvectors.conj().T
    frames_energy = jnp.matmul(jnp.matmul(adjoint[None, :, :], frames), eigenvectors[None, :, :])
    target_energy = adjoint @ target_state @ eigenvectors
    gaps = energies[:, None] - energies[None, :]
    multiplier = -2j * jnp.tanh(0.5 * beta * gaps)
    scores = multiplier[None, :, :] * frames_energy
    loss_terms = (
        0.5 * jnp.matmul(scores, scores)
        - 1j * (jnp.matmul(frames_energy, scores) - jnp.matmul(scores, frames_energy))
    )
    loss_observable = jnp.sum(loss_terms, axis=0)
    return jnp.real(jnp.trace(target_energy @ loss_observable))


loss_and_gradient = jax.jit(jax.value_and_grad(qsm_loss, argnums=0))
loss_only = jax.jit(qsm_loss)
batched_loss_and_gradient = jax.jit(
    jax.vmap(jax.value_and_grad(qsm_loss, argnums=0), in_axes=(0, None, None, None, None)),
)


def make_problem(n: int, beta: float, model_name: str = "local_tfim") -> ExactQSMProblem:
    terms_np, frames_np = build_dense_model(n, model_name)
    terms = jnp.asarray(terms_np, dtype=COMPLEX)
    frames = jnp.asarray(frames_np, dtype=COMPLEX)
    theta_star = target_theta(n, model_name)
    target_hamiltonian = jnp.tensordot(jnp.asarray(theta_star, dtype=REAL), terms, axes=1)
    target_state = gibbs_state(target_hamiltonian, beta)
    target_loss = float(loss_only(jnp.asarray(theta_star, dtype=REAL), terms, frames, target_state, beta))
    gram_diagonal = np.asarray([8.0] * (n - 1) + [4.0] * n)
    return ExactQSMProblem(
        model_name=model_name,
        n=n,
        beta=beta,
        terms=terms,
        frames=frames,
        theta_star=theta_star,
        target_state=target_state,
        target_loss=target_loss,
        gram_diagonal=gram_diagonal,
    )


def population_point(problem: ExactQSMProblem, theta: np.ndarray) -> tuple[float, np.ndarray]:
    value, gradient = loss_and_gradient(
        jnp.asarray(theta, dtype=REAL),
        problem.terms,
        problem.frames,
        problem.target_state,
        problem.beta,
    )
    return float(value), np.asarray(gradient, dtype=float)


def population_points(problem: ExactQSMProblem, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, gradients = batched_loss_and_gradient(
        jnp.asarray(theta, dtype=REAL),
        problem.terms,
        problem.frames,
        problem.target_state,
        problem.beta,
    )
    return np.asarray(values, dtype=float), np.asarray(gradients, dtype=float)


__all__ = [
    "ExactQSMProblem",
    "loss_and_gradient",
    "make_problem",
    "population_point",
    "population_points",
    "qsm_loss",
    "target_theta",
]
