#include <rhoban_random/gaussian_mixture_model.h>

#include <rhoban_random/tools.h>

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
GaussianMixtureModel::~GaussianMixtureModel()
{
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
    likelihood += exp(log(gaussians_weights[i] / getTotalWeight()) + gaussians[i].getLogLikelihood(point));
  }
  return log(likelihood);
}
double GaussianMixtureModel::getLogLikelihood(const Eigen::MatrixXd& points) const
{
  double result = 0.0;
  for (int idx = 0; idx < points.cols(); idx++)
  {
    const Eigen::VectorXd& point = points.col(idx);
    result += getLogLikelihood(point);
  }
  return result;
}

std::string GaussianMixtureModel::getClassName() const
{
  return "GaussianMixtureModel";
}

void GaussianMixtureModel::fromJson(const Json::Value& v, const std::string& dir_name)
{
  std::vector<MultivariateGaussian> new_gaussians = rhoban_utils::readVector<MultivariateGaussian>(
      v, "gaussians", dir_name, [](const Json::Value& v, const std::string& dir_name) {
        MultivariateGaussian g;
        g.fromJson(v, dir_name);
        return g;
      });
  std::vector<double> new_gaussians_weights = rhoban_utils::readVector<double>(v, "gaussians_weights");
  if (new_gaussians.size() != new_gaussians_weights.size())
  {
    throw rhoban_utils::JsonParsingError("Mismatch of size between 'gaussians' and 'gaussians weights'");
  }
  gaussians.clear();
  gaussians_weights.clear();
  for (size_t idx = 0; idx < new_gaussians.size(); idx++)
  {
    addGaussian(new_gaussians[idx], new_gaussians_weights[idx]);
  }
}

Json::Value GaussianMixtureModel::toJson() const
{
  Json::Value v;
  for (size_t idx = 0; idx < size(); idx++)
  {
    v["gaussians"].append(gaussians[idx].toJson());
    v["gaussians_weights"].append(gaussians_weights[idx]);
  }
  return v;
}

}  // namespace rhoban_random
