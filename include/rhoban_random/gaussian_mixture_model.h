#pragma once

#include <rhoban_random/multivariate_gaussian.h>

namespace rhoban_random
{
/// A model in which multiple MultivariateGaussian are used, each one with its own probability
class GaussianMixtureModel
{
public:
  GaussianMixtureModel();
  GaussianMixtureModel(const std::vector<MultivariateGaussian>& gaussians, const std::vector<double>& weights);

  /// Returns the number of gaussians
  size_t size() const;
  /// The number of dimensions of the mixture model.
  /// - 0 if empty
  size_t dimension() const;

  const MultivariateGaussian& getGaussian(size_t idx) const;
  double getGaussianWeight(size_t idx) const;

  /// Return the sum of the weights of all the gaussians
  double getTotalWeight() const;

  void addGaussian(const MultivariateGaussian& gaussian, double weight);

  /// Sample a point from the multivariate gaussian with given random engine
  Eigen::VectorXd getSample(std::default_random_engine* engine) const;

  /// Sample multiple points from the multivariate gaussian with given random engine
  /// Each column is a different point
  Eigen::MatrixXd getSamples(int nb_samples, std::default_random_engine* engine) const;

  /// Return the density of probability at 'point' given the distribution
  /// parameters.
  /// This method is very likely to return extremely small numbers, therefore
  /// it is preferable to use getLogLikelihood
  double getLikelihood(const Eigen::VectorXd& point) const;

  /// Return the logarithm of the likelihood at the given point
  double getLogLikelihood(const Eigen::VectorXd& point) const;

  /// Constraints on covariance Matrix for EM algorithm
  enum EMCovMatType
  {
    /// Cov Matrix is a scaled Identity matrix ($\alpha I$ with $\alpha the only free parameter)
    COV_MAT_SPHERICAL,
    /// Cov Matrix is a diagonal matrix with one free parameter per dimension
    COV_MAT_DIAGONAL,
    /// Cov Matrix is symmetric positively defined, order of nb parameters: d^2/2
    /// TODO: refine the number of parameters implied
    COV_MAT_GENERIC
  };

  /// Uses the ExpectationMaximization algorithm to obtain a gaussian mixture model based on the provided points
  /// - n: number of clusters
  /// - inputs: each column is a different point
  /// - labels: if not null, is filled with the id of the gaussian attributed to each element
  /// - max_iterations: the maximal number of iteration for the EM algorithm
  /// - epsilon: if improvement of logLikelihood goes below this value, stop iterations
  static GaussianMixtureModel EM(int n, const Eigen::MatrixXd& inputs, EMCovMatType cov_mat_type,
                                 Eigen::VectorXi* labels = nullptr, size_t max_iterations = 100,
                                 double epsilon = std::pow(10, -6));

private:
  std::vector<MultivariateGaussian> gaussians;
  std::vector<double> gaussians_weights;
};
}  // namespace rhoban_random
