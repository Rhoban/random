# Rhoban Random

A library allowing to sample classical distributions more easily but also
allowing to measure the probability density function for basic distributions.


## Gaussian mixture model

This library uses `OpenCV` implementation of the Expectation-Maximization
algorithm to build Gaussian Mixture Models from data. By using the Bayesian
Information Criterion (BIC), it can automatically choose an approriate number of
clusters.

Results can be compared with python `sklearn` implementation to ensure
implementation is correct. The different tools are as follows:

- `examples/generate_gmm_samples.cpp`: A tool allowing to produce samples in a
  `csv` file based on a `gaussian mixture model` described in a `json` file. An
  example of `json` file can be found in `configs/gmm.json`.
- `examples/expectation_maximization.cpp`: A tool allowing to compare the effect
  of different numbers of gaussians and different types of parametrization on both
  BIC and AIC.
- `external/gmm.py`: An `sklearn` based implementation modified to take a `csv`
  file as input.
