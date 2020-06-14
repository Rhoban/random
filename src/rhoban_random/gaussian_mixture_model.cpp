#include <rhoban_random/gaussian_mixture_model.h>

#include <rhoban_random/tools.h>

// #include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/ml.hpp>

namespace rhoban_random
{
GaussianMixtureModel::GaussianMixtureModel()
{
}
GaussianMixtureModel::GaussianMixtureModel(const std::vector<MultivariateGaussian>& new_gaussians,
                                           const std::vector<double>& new_weights)
{
  if (new_gaussians.size() != new_weights.size())
    throw std::logic_error("gaussians size and weights size mismatch");
  for (int i = 0; i < new_gaussians.size(); i++)
  {
    addGaussian(new_gaussians[i], new_weights[i]);
  }
}

size_t GaussianMixtureModel::size() const
{
  return gaussians.size();
}

size_t GaussianMixtureModel::dimension() const
{
  if (size() == 0)
    return 0;
  return gaussians[0].dimension();
}

const MultivariateGaussian& GaussianMixtureModel::getGaussian(size_t idx) const
{
  return gaussians[idx];
}

double GaussianMixtureModel::getGaussianWeight(size_t idx) const
{
  return gaussians_weights[idx];
}

double GaussianMixtureModel::getTotalWeight() const
{
  double sum = 0;
  for (double w : gaussians_weights)
    sum += w;
  return sum;
}

void GaussianMixtureModel::addGaussian(const MultivariateGaussian& gaussian, double weight)
{
  if (size() > 0)
  {
    size_t expected_dimension = getGaussian(0).dimension();
    size_t received_dimension = gaussian.dimension();
    if (expected_dimension != received_dimension)
    {
      throw std::logic_error("dimension mismatch: received " + std::to_string(received_dimension) +
                             ", while expecting " + std::to_string(expected_dimension));
    }
  }
  if (weight <= 0)
  {
    throw std::logic_error("weights should be strictly positive, received: " + std::to_string(weight));
  }
  gaussians.push_back(gaussian);
  gaussians_weights.push_back(weight);
}

Eigen::VectorXd GaussianMixtureModel::getSample(std::default_random_engine* engine) const
{
  int idx = sampleWeightedIndices(gaussians_weights, 1, engine)[0];
  return gaussians[idx].getSample(engine);
}

Eigen::MatrixXd GaussianMixtureModel::getSamples(int nb_samples, std::default_random_engine* engine) const
{
  int D = dimension();
  int idx = 0;
  Eigen::MatrixXd result(D, nb_samples);
  std::map<int, int> samples = sampleWeightedIndicesMap(gaussians_weights, nb_samples, engine);
  for (const std::pair<int, int>& entry : samples)
  {
    int gaussian_idx = entry.first;
    int gaussian_samples = entry.second;
    result.block(0, idx, D, gaussian_samples) = getGaussian(gaussian_idx).getSamples(gaussian_samples, engine);
    idx += gaussian_samples;
  }
  return result;
}

double GaussianMixtureModel::getLikelihood(const Eigen::VectorXd& point) const
{
  if (size() == 0)
    throw std::logic_error("Empty GaussianMixtureModel");
  double likelihood = 0;
  for (int i = 0; i < size(); i++)
  {
    likelihood += gaussians_weights[i] * gaussians[i].getLikelihood(point);
  }
  return likelihood / getTotalWeight();
}

double GaussianMixtureModel::getLogLikelihood(const Eigen::VectorXd& point) const
{
  if (size() == 0)
    throw std::logic_error("Empty GaussianMixtureModel");
  double likelihood = 0;
  for (int i = 0; i < size(); i++)
  {
    likelihood += exp(log(gaussians_weights[i]) + gaussians[i].getLogLikelihood(point));
  }
  return log(likelihood / getTotalWeight());
}

cv::ml::EM::Types getCVType(GaussianMixtureModel::EMCovMatType type)
{
  switch (type)
  {
    case GaussianMixtureModel::EMCovMatType::COV_MAT_SPHERICAL:
      return cv::ml::EM::COV_MAT_SPHERICAL;
    case GaussianMixtureModel::EMCovMatType::COV_MAT_DIAGONAL:
      return cv::ml::EM::COV_MAT_DIAGONAL;
    case GaussianMixtureModel::EMCovMatType::COV_MAT_GENERIC:
      return cv::ml::EM::COV_MAT_GENERIC;
    default:
      throw std::logic_error("Unknown type");
  }
}

GaussianMixtureModel GaussianMixtureModel::EM(int n, const Eigen::MatrixXd& inputs, EMCovMatType cov_mat_type,
                                              Eigen::VectorXi* labels, size_t max_iterations, double epsilon)
{
  int nb_points = inputs.cols();
  int D = inputs.rows();
  std::cout << "Starting EM with: " << std::endl
            << "\tNb points: " << nb_points << std::endl
            << "\tDim: " << D << std::endl
            << "\tNb gaussians wished: " << n << std::endl;
  cv::Mat cv_points;
  Eigen::MatrixXd points = inputs.transpose();
  cv::eigen2cv(points, cv_points);
  cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
  em_model->setClustersNumber(n);
  em_model->setCovarianceMatrixType(getCVType(cov_mat_type));
  em_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, max_iterations, epsilon));
  cv::Mat cv_labels = cv::Mat::zeros(nb_points, 1, CV_32S);
  em_model->trainEM(cv_points, cv::noArray(), cv_labels, cv::noArray());
  /// Importing result
  std::vector<cv::Mat> covs;
  em_model->getCovs(covs);
  cv::Mat means = em_model->getMeans();
  cv::Mat weights = em_model->getWeights();
  std::cout << "Means size: " << means.size() << std::endl;
  GaussianMixtureModel result;
  for (int i = 0; i < n; i++)
  {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covar;
    double weight;
    cv::cv2eigen(means.row(i).t(), mean);
    cv::cv2eigen(covs[i], covar);
    weight = weights.at<double>(i);
    std::cout << "Adding a gaussian with: " << std::endl
              << "mean: " << mean.transpose() << std::endl
              << "covar:" << std::endl
              << covar << std::endl
              << "weight: " << weight << std::endl;
    result.addGaussian(MultivariateGaussian(mean, covar), weight);
  }
  return result;
}

}  // namespace rhoban_random
