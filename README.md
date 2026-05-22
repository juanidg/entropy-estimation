# entropy-estimation
Python implementations of the Correlation Coverage-Adjusted (CC) and Corrected Miller–Madow (CMM) entropy estimators for discrete sequences.

# Entropy Estimators for Discrete Sequences

Python implementations of two entropy estimators designed for correlated discrete sequences:

- Correlation Coverage-Adjusted estimator (CC): estimator designed to account for correlations and unseen states in finite samples.
- Corrected Miller–Madow estimator (CMM): extension of the Miller–Madow estimator including a correction term based on the eigenvalues of the empirical transition matrix.

These implementations are adapted for blocks of size 1 (ordinary discrete entropy estimation) and support arbitrary discrete alphabets.

# Reference
This repository accompanies the paper:

De Gregorio, J., Sánchez, D., & Toral, R. (2024). Entropy Estimators for Markovian Sequences: A Comparative Analysis. Entropy 26(1), 79.
