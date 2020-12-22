#pragma once

#include <rhoban_random/multivariate_gaussian.h>
#include "rhoban_utils/serialization/json_serializable.h"

namespace rhoban_random
{
/// A model in which multiple MultivariateGaussian are used, each one with its own probability
class GaussianMixtureModel : public rhoban_utils::JsonSerializable
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

  /// Get posterior vector (the size will be the same as the number of gaussians)
  Eigen::VectorXd getPosteriors(const Eigen::VectorXd& point) const;

  /// Get the gradient of the likelihood for a given point
  Eigen::VectorXd likelihoodGradient(const Eigen::VectorXd& point) const;

  /// Get the hessian of the likelihood for a given point
  Eigen::MatrixXd likelihoodHessian(const Eigen::VectorXd& point) const;

  /// Deserializes from a json content found in 'dir_name'
  void fromJson(const Json::Value& json_value, const std::string& dir_name);

  /// Represent current object as a Json::Value
  Json::Value toJson() const;

  std::string getClassName() const;

  /// Remaps (reorder) the variables according to given remapping
  /// This remapping should be an array containing all indexes once
  void remap(const std::vector<int>& remapping);

  /// Equivalent to remap the variables backward
  void remap_invert();

  /// Marginalize the last dimensions (keep only n variables)
  GaussianMixtureModel marginalize(int n) const;

  /// Condition the GMM with a given sample
  GaussianMixtureModel condition(const Eigen::VectorXd& value) const;

  int n_parameters() const;
  double bic(const std::vector<Eigen::VectorXd>& points) const;

#ifdef ENABLE_OPENCV
  /// Using (OpenCV) EM to fit given points
  static GaussianMixtureModel fit(const std::vector<Eigen::VectorXd>& points, int minClusters = 1,
                                  int maxClusters = 16);
#endif

private:
  std::vector<MultivariateGaussian> gaussians;
  std::vector<double> gaussians_weights;
};
}  // namespace rhoban_random
