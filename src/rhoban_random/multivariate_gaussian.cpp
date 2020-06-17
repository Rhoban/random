#include "rhoban_random/multivariate_gaussian.h"

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>

namespace rhoban_random
{
MultivariateGaussian::MultivariateGaussian() : mu(), covar(), cholesky(), determinant(0.0)
{
}

MultivariateGaussian::MultivariateGaussian(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance)
  : mu(mean), covar(covariance), cholesky(), determinant(0.0)
{
  // Check size
  if (mean.size() != covariance.rows() || mean.size() != covariance.cols())
  {
    throw std::logic_error("MultivariateGaussian: invalid input size: mean is " + std::to_string(mean.size()) +
                           " and covar is " + std::to_string(covariance.rows()) + "x" +
                           std::to_string(covariance.cols()));
  }

  // Cholesky decomposition of covariance matrix
  computeDecomposition();
}

MultivariateGaussian::~MultivariateGaussian()
{
}

size_t MultivariateGaussian::dimension() const
{
  return mu.size();
}

const Eigen::VectorXd& MultivariateGaussian::getMean() const
{
  return mu;
}
const Eigen::MatrixXd& MultivariateGaussian::getCovariance() const
{
  return covar;
}

Eigen::VectorXd MultivariateGaussian::getSample(std::default_random_engine* engine) const
{
  // Draw normal unit vector
  Eigen::VectorXd unitRand(mu.size());
  std::normal_distribution<double> dist(0.0, 1.0);
  for (size_t i = 0; i < (size_t)unitRand.size(); i++)
  {
    unitRand(i) = dist(*engine);
  }

  // Compute the random generated point
  Eigen::VectorXd point = mu + cholesky * unitRand;
  return point;
}

Eigen::MatrixXd MultivariateGaussian::getSamples(int nb_samples, std::default_random_engine* engine) const
{
  Eigen::MatrixXd result(mu.rows(), nb_samples);
  for (int i = 0; i < nb_samples; i++)
  {
    result.col(i) = getSample(engine);
  }
  return result;
}

double MultivariateGaussian::getLikelihood(const Eigen::VectorXd& point) const
{
  size_t size = mu.size();
  // Compute distance from mean
  Eigen::VectorXd delta = computeDistanceFromMean(point);
  // throw error if point is out of range
  if (delta.size() == 0)
  {
    throw std::logic_error("MultivariateGaussian: point is out of range");
  }

  // Compute likelihood
  double tmp1 = -0.5 * delta.transpose() * covar_inv * delta;
  double tmp2 = pow(2.0 * M_PI, size) * determinant;
  return std::exp(tmp1) / std::sqrt(tmp2);
}

double MultivariateGaussian::getLogLikelihood(const Eigen::VectorXd& point) const
{
  size_t size = mu.size();
  // Compute distance from mean
  Eigen::VectorXd delta = computeDistanceFromMean(point);
  // throw error if point is out of range
  if (delta.size() == 0)
  {
    throw std::logic_error("MultivariateGaussian: point is out of range");
  }

  // Compute log likelihood
  double tmp1 = delta.transpose() * covar_inv * delta;
  return -0.5 * (std::log(determinant) + tmp1 + (double)size * std::log(2.0 * M_PI));
}

void MultivariateGaussian::fit(const std::vector<Eigen::VectorXd>& data)
{
  // Check sizes
  if (data.size() < 2)
  {
    throw std::logic_error("MultivariateGaussian::fit: not enough data points");
  }
  size_t size = data.front().size();
  for (size_t i = 0; i < data.size(); i++)
  {
    if ((size_t)data[i].size() != size)
    {
      throw std::logic_error("MultivariateGaussian::fit: invalid data dimension");
    }
  }

  Eigen::VectorXd sum = Eigen::VectorXd::Zero(size);
  for (size_t i = 0; i < data.size(); i++)
  {
    sum += data[i];
  }
  mu = (1.0 / (double)data.size()) * sum;

  // Compute the covariance estimation
  Eigen::MatrixXd sum2 = Eigen::MatrixXd::Zero(size, size);
  for (size_t i = 0; i < data.size(); i++)
  {
    Eigen::VectorXd delta = computeDistanceFromMean(data[i]);
    sum2 += delta * delta.transpose();
  }
  covar = (1.0 / (double)(data.size() - 1)) * sum2;

  // Update the Cholesky decomposition
  computeDecomposition();
}

void MultivariateGaussian::computeDecomposition()
{
  // Add small epsilon on diagonal according
  // to Rasmussen 2006
  double epsilon = 1e-9;
  size_t size = mu.size();
  Eigen::MatrixXd noise = epsilon * Eigen::MatrixXd::Identity(size, size);
  Eigen::LLT<Eigen::MatrixXd> llt(covar + noise);
  cholesky = llt.matrixL();
  // Check the decomposition
  if (llt.info() == Eigen::NumericalIssue)
  {
    throw std::logic_error("MultivariateGaussian Cholesky decomposition error");
  }
  // Compute the covariance determinant
  determinant = 1.0;
  for (size_t i = 0; i < (size_t)mu.size(); i++)
  {
    determinant *= cholesky(i, i);
  }
  determinant = pow(determinant, 2);
  // Compute the covariance inverse
  covar_inv = llt.solve(Eigen::MatrixXd::Identity(size, size));
}

Eigen::VectorXd MultivariateGaussian::computeDistanceFromMean(const Eigen::VectorXd& point) const
{
  size_t size = mu.size();
  if ((size_t)point.size() != size)
  {
    throw std::logic_error("MultivariateGaussian: invalid dimension");
  }

  Eigen::VectorXd delta = point - mu;

  return delta;
}  // namespace rhoban_random

std::string MultivariateGaussian::getClassName() const
{
  return "MultivariateGaussian";
}
void MultivariateGaussian::fromJson(const Json::Value& v, const std::string& dir_name)
{
  rhoban_utils::tryReadEigen(v, "mu", &mu);
  rhoban_utils::tryReadEigen(v, "covar", &covar);
  computeDecomposition();
}

Json::Value MultivariateGaussian::toJson() const
{
  Json::Value v;
  v["mu"] = rhoban_utils::vector2Json(mu);
  v["covar"] = rhoban_utils::matrix2Json(mu);
  return v;
}

}  // namespace rhoban_random
