# Monte Carlo Dropout — Uncertainty Estimation

This folder contains the Jupyter notebooks implementing **Monte Carlo Dropout** to estimate uncertainty in Whisper transcriptions.

## Contents

- **Main experiment notebook**  
  Implements:
  - Whisper model loading
  - Dropout activation at inference time
  - Multiple stochastic forward passes
  - Variance and entropy–based uncertainty scores
  - Correlation analysis
  - WER computation
  - Visualization of uncertainty vs. error
