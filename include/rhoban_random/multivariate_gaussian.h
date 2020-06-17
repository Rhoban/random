/// This file was imported from Quentin Rouxel PhD thesis repository
/// with some style modifications
#pragma once

#include <vector>
#include <random>
#include <Eigen/Core>

#include <rhoban_utils/serialization/json_serializable.h>

namespace rhoban_random
{
/// This class implement a Multivariate Gaussian distribution. It provides
/// access to measures such as the density of probability at a given point
class MultivariateGaussian : public rhoban_utils::JsonSerializable
{
public:
  /// Dummy empty initialization
  MultivariateGaussian();

  /// Initialization with mean vector and covariance matrix.
  MultivariateGaussian(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance);
  ~MultivariateGaussian();

  /// Return the gaussian
  /// dimentionality
  size_t dimension() const;

  const Eigen::VectorXd& getMean() const;
  const Eigen::MatrixXd& getCovariance() const;

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

  /// Compute the classic estimation of gaussian mean and covariance from given
  /// data vectors.
  void fit(const std::vector<Eigen::VectorXd>& points);

  std::string getClassName() const override;
  void fromJson(const Json::Value& value, const std::string& dir_name) override;
  Json::Value toJson() const override;

private:
  /// The mean vector
  Eigen::VectorXd mu;

  /// The covariance matrix (symetrix definite positive)
  Eigen::MatrixXd covar;

  /// The inverse of covariance matrix computed through cholesky decomposition
  Eigen::MatrixXd covar_inv;

  /// The left side of the Cholesky decomposition of the covariance matrix
  Eigen::MatrixXd cholesky;

  /// The determinant of the covariance matrix
  double determinant;

  /// Compute the covariance decomposition and update internal variables
  void computeDecomposition();

  /// Return the signed distance between
  /// given point and current mean.
  /// @throw logic_error if dimension of the point does not match dimension of
  ///        the distribution
  Eigen::VectorXd computeDistanceFromMean(const Eigen::VectorXd& point) const;
};

}  // namespace rhoban_random
