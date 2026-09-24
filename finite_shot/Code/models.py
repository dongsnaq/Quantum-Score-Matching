"""Dense Pauli strings for the inhomogeneous TFIM example."""
from __future__ import annotations
import numpy as np

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1., -1.]).astype(complex),
}


def pauli_string(n: int, operators: dict[int, str]) -> np.ndarray:
    """Return a Pauli string with site zero as the leftmost tensor factor."""
    result = np.array([[1.0]], dtype=complex)
    for site in range(n):
        result = np.kron(result, PAULI[operators.get(site, "I")])
    return result
