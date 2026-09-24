from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import zeta


C_U = 7.0 * zeta(3.0, 1.0) / np.pi**3


def absolute_scaled_time_density(x: float) -> float:
    if x == 0.0:
        return np.inf
    return float(-(4.0 / np.pi) * np.log(np.tanh(0.5 * np.pi * x)))


@lru_cache(maxsize=512)
def scaled_tail_moments(x: float) -> tuple[float, float, float]:
    probability = quad(
        absolute_scaled_time_density, x, np.inf, epsabs=1e-12, epsrel=1e-10
    )[0]
    first = quad(
        lambda y: y * absolute_scaled_time_density(y),
        x,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-10,
    )[0]
    second = quad(
        lambda y: y * y * absolute_scaled_time_density(y),
        x,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-10,
    )[0]
    return float(probability), float(first), float(second)


def tail_bias_bound(beta: float, scaled_cutoff: float, b: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_tail, scaled_first_tail, _ = scaled_tail_moments(scaled_cutoff)
    first_tail = beta * scaled_first_tail
    score_tail = beta * b * p_tail
    response_tail = beta * q * p_tail + 2.0 * beta * b[:, None] * first_tail
    full_response = beta * q + 2.0 * C_U * beta**2 * b[:, None]
    retained_score_plus_frame = beta * b * (1.0 - p_tail) + 2.0
    return np.sum(
        score_tail[:, None] * full_response
        + retained_score_plus_frame[:, None] * response_tail,
        axis=0,
    )


def uniform_cutoff(beta: float, target_bias: float, b: np.ndarray, q: np.ndarray) -> float:
    lower, upper = 1.0, 2.0
    while float(np.max(tail_bias_bound(beta, upper, b, q))) > target_bias:
        lower, upper = upper, 2.0 * upper
        if upper > 256.0:
            raise RuntimeError("failed to certify a finite time cutoff")
    for _ in range(55):
        midpoint = 0.5 * (lower + upper)
        if float(np.max(tail_bias_bound(beta, midpoint, b, q))) <= target_bias:
            upper = midpoint
        else:
            lower = midpoint
    return beta * upper


def truncated_resource_data(
    beta: float, cutoff: float, b: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scaled_cutoff = cutoff / beta
    p_tail, scaled_first_tail, scaled_second_tail = scaled_tail_moments(scaled_cutoff)
    retained_probability = 1.0 - p_tail
    retained_first = beta * (C_U - scaled_first_tail)
    retained_second = beta**2 * (1.0 / 6.0 - scaled_second_tail)
    score_mass = beta * b * retained_probability + 2.0
    response_mass = beta * q * retained_probability + 2.0 * beta * b[:, None] * retained_first
    coefficient_mass = np.sum(score_mass[:, None] * response_mass, axis=0)
    score_time_mass = beta * b * retained_first
    response_time_mass = beta * q * retained_first + 2.0 * beta * b[:, None] * retained_second
    time_weighted_mass = np.sum(
        score_time_mass[:, None] * response_mass
        + score_mass[:, None] * response_time_mass,
        axis=0,
    )
    mean_time = np.divide(
        time_weighted_mass,
        coefficient_mass,
        out=np.zeros_like(coefficient_mass),
        where=coefficient_mass > 0,
    )
    return coefficient_mass, mean_time
