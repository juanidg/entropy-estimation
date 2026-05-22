#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entropy estimators for discrete sequences.

Implemented estimators:
    - Correlation Coverage-Adjusted estimator (CC)
    - Corrected Miller-Madow estimator (CMM)

Both estimators are implemented for blocks of size 1 only.

Reference:
De Gregorio, Sánchez, Toral (2024)
"Entropy Estimators for Markovian Sequences:
A Comparative Analysis"
"""

import numpy as np
from collections import Counter, defaultdict


def _log(x, base=np.e):
    """
    Logarithm with arbitrary base.
    """
    return np.log(x) / np.log(base)


def mle_entropy(sequence, base=np.e):
    """
    Maximum likelihood (plug-in) entropy estimator.

    Parameters
    ----------
    sequence : array-like
        Sequence of discrete symbols.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Estimated entropy.
    """

    sequence = np.asarray(sequence)
    N = len(sequence)

    counts = np.array(list(Counter(sequence).values()))
    probs = counts / N

    return -np.sum(probs * _log(probs, base))


def correlation_coverage_adjusted_entropy(sequence, base=np.e):
    """
    Correlation Coverage-Adjusted (CC) entropy estimator.

    Parameters
    ----------
    sequence : array-like
        Sequence of discrete symbols.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Estimated entropy.
    """

    sequence = np.asarray(sequence)
    N = len(sequence)

    if N == 0:
        raise ValueError("Sequence must contain at least one element.")

    counts = Counter(sequence)

    # Empirical probabilities
    probs = np.array(list(counts.values()), dtype=float) / N

    # Sequential coverage correction
    N_half = N // 2

    observed = set(sequence[:N_half])

    coverage = 1.0

    for j in range(N_half, N):

        if sequence[j] not in observed:
            coverage -= 1.0 / (j + 1)

        observed.add(sequence[j])

    corrected_probs = coverage * probs

    # Avoid numerical issues
    corrected_probs = corrected_probs[corrected_probs > 0]

    entropy = -np.sum(
        corrected_probs
        * _log(corrected_probs, base)
        / (1 - (1 - corrected_probs) ** N)
    )

    return entropy


def corrected_miller_madow_entropy(sequence, base=np.e):
    """
    Corrected Miller-Madow (CMM) entropy estimator.

    Parameters
    ----------
    sequence : array-like
        Sequence of discrete symbols.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Estimated entropy.
    """

    sequence = np.asarray(sequence)
    N = len(sequence)

    if N < 2:
        raise ValueError("Sequence must contain at least two elements.")

    # ------------------------------------------------------------------
    # State counts
    # ------------------------------------------------------------------

    counts = Counter(sequence)

    states = sorted(counts.keys())
    state_to_index = {s: i for i, s in enumerate(states)}

    counts_array = np.array([counts[s] for s in states], dtype=float)

    probs = counts_array / N

    # ------------------------------------------------------------------
    # Miller-Madow estimator
    # ------------------------------------------------------------------

    H_mle = -np.sum(probs * _log(probs, base))

    N0 = len(states)

    H_mm = H_mle + (N0 - 1) / (2 * N)

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    L = len(states)

    transition_counts = np.zeros((L, L), dtype=float)

    for a, b in zip(sequence[:-1], sequence[1:]):

        i = state_to_index[a]
        j = state_to_index[b]

        transition_counts[i, j] += 1

    transition_matrix = np.zeros((L, L), dtype=float)

    for i in range(L):

        row_sum = np.sum(transition_counts[i])

        if row_sum > 0:
            transition_matrix[i] = transition_counts[i] / row_sum

        else:
            # Fallback:
            # use empirical stationary probabilities
            transition_matrix[i] = probs

    # ------------------------------------------------------------------
    # Eigenvalue correction
    # ------------------------------------------------------------------

    eigenvalues = np.linalg.eigvals(transition_matrix)

    # Remove eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    eigenvalues = np.delete(eigenvalues, idx)

    correction = 0.0

    for lam in eigenvalues:

        # Avoid divergences near |lambda| = 1
        if np.abs(lam) < 1 - 1e-12:

            correction += (lam / (1 - lam)).real

    H_cmm = H_mm + correction / N

    return H_cmm


if __name__ == "__main__":

    rng = np.random.default_rng(42)

    # Example sequence
    sequence = rng.integers(0, 2, size=100)

    print("MLE:", mle_entropy(sequence))
    print("CC :", correlation_coverage_adjusted_entropy(sequence))
    print("CMM:", corrected_miller_madow_entropy(sequence))
