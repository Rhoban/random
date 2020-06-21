#pragma once

#include <rhoban_random/multivariate_gaussian.h>

namespace rhoban_random
{
/// A model in which multiple MultivariateGaussian are used, each one with its own probability
class GaussianMixtureModel : public rhoban_utils::JsonSerializable
{
public:
  GaussianMixtureModel();
  GaussianMixtureModel(const std::vector<MultivariateGaussian>& gaussians, const std::vector<double>& weights);
  virtual ~GaussianMixtureModel();

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

  /// Return the log-likelihood of the given set of point (each column is a different point)
  double getLogLikelihood(const Eigen::MatrixXd& points) const;

  std::string getClassName() const override;
  void fromJson(const Json::Value& value, const std::string& dir_name) override;
  Json::Value toJson() const override;

private:
  std::vector<MultivariateGaussian> gaussians;
  std::vector<double> gaussians_weights;
};
}  // namespace rhoban_random
