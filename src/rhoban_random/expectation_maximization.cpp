#include <rhoban_random/expectation_maximization.h>

#include <rhoban_utils/util.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/ml.hpp>

namespace rhoban_random
{
ExpectationMaximization::ExpectationMaximization()
  : min_clusters(1)
  , max_clusters(10)
  , cov_mat_types_allowed({ CovMatType::Spherical, CovMatType::Diagonal, CovMatType::Generic })
  , max_iterations(100)
  , epsilon(std::pow(10, -6))
{
}

void ExpectationMaximization::setMinClusters(size_t new_min)
{
  min_clusters = new_min;
}

void ExpectationMaximization::setMaxClusters(size_t new_max)
{
  max_clusters = new_max;
}

void ExpectationMaximization::setAllowedCovMatTypes(const std::set<CovMatType>& allowed_types)
{
  cov_mat_types_allowed = allowed_types;
}

void ExpectationMaximization::analyze(const Eigen::MatrixXd& data)
{
  setPoints(data);
  for (size_t nb_clusters = min_clusters; nb_clusters <= max_clusters; nb_clusters++)
  {
    for (CovMatType type : cov_mat_types_allowed)
    {
      entries[{ nb_clusters, type }] = run(nb_clusters, type);
    }
  }
}

ExpectationMaximization::Configuration ExpectationMaximization::getBestConfiguration(ScoreCriterion criterion) const
{
  if (entries.size() == 0)
    throw std::out_of_range(DEBUG_INFO + "no entries available");
  double best_score = std::numeric_limits<double>::max();
  Configuration best_configuration;
  for (const auto& entry : entries)
  {
    double entry_score = entry.second.scores.at(criterion);
    if (entry_score < best_score)
    {
      best_score = entry_score;
      best_configuration = entry.first;
    }
  }
  return best_configuration;
}

const ExpectationMaximization::Result& ExpectationMaximization::getBestResult(ScoreCriterion criterion) const
{
  return entries.at(getBestConfiguration(criterion));
}

const ExpectationMaximization::Result& ExpectationMaximization::getResult(const Configuration& c) const
{
  return entries.at(c);
}

rhoban_utils::StringTable ExpectationMaximization::getScoresTable() const
{
  rhoban_utils::StringTable result({ "nbClusters", "covMatType", "logLikelihood", "AIC", "BIC" });
  for (const auto& entry : entries)
  {
    result.insertRow({ { "nbClusters", std::to_string(entry.first.first) },
                       { "covMatType", toString(entry.first.second) },
                       { "logLikelihood", std::to_string(entry.second.scores.at(LogLikelihood)) },
                       { "AIC", std::to_string(entry.second.scores.at(AIC)) },
                       { "BIC", std::to_string(entry.second.scores.at(BIC)) } });
  }
  return result;
}
std::string ExpectationMaximization::toString(rhoban_random::ExpectationMaximization::CovMatType type)
{
  switch (type)
  {
    case rhoban_random::ExpectationMaximization::CovMatType::Spherical:
      return "Spherical";
    case rhoban_random::ExpectationMaximization::CovMatType::Diagonal:
      return "Diagonal";
    case rhoban_random::ExpectationMaximization::CovMatType::Generic:
      return "Generic";
    default:
      throw std::logic_error(DEBUG_INFO + "Unknown type" + std::to_string(type));
  }
}

void ExpectationMaximization::setPoints(const Eigen::MatrixXd& new_points)
{
  entries.clear();
  points = new_points;
}

size_t ExpectationMaximization::getNbParameters(size_t dimensions, size_t nb_clusters, CovMatType cov_mat_type)
{
  size_t nb_gaussian_weights = nb_clusters - 1;
  size_t nb_gaussian_means = nb_clusters * dimensions;
  size_t nb_covar_params = 0;
  switch (cov_mat_type)
  {
    case CovMatType::Spherical:
      nb_covar_params = nb_clusters;
      break;
    case CovMatType::Diagonal:
      nb_covar_params = dimensions * nb_clusters;
      break;
    case CovMatType::Generic:
      nb_covar_params = (dimensions * (dimensions + 1) / 2) * nb_clusters;
      break;
    default:
      throw std::logic_error(DEBUG_INFO + "unkown CovMatType");
  }
  return nb_gaussian_means + nb_gaussian_weights + nb_covar_params;
}

cv::ml::EM::Types getCVType(ExpectationMaximization::CovMatType type)
{
  switch (type)
  {
    case ExpectationMaximization::CovMatType::Spherical:
      return cv::ml::EM::COV_MAT_SPHERICAL;
    case ExpectationMaximization::CovMatType::Diagonal:
      return cv::ml::EM::COV_MAT_DIAGONAL;
    case ExpectationMaximization::CovMatType::Generic:
      return cv::ml::EM::COV_MAT_GENERIC;
    default:
      throw std::logic_error("Unknown type");
  }
}

ExpectationMaximization::Result ExpectationMaximization::run(int n, CovMatType cov_mat_type)
{
  int nb_points = points.cols();
  int D = points.rows();
  // std::cout << "Starting EM with: " << std::endl
  //           << "\tNb points: " << nb_points << std::endl
  //           << "\tDim: " << D << std::endl
  //           << "\tNb gaussians wished: " << n << std::endl;
  cv::Mat cv_points;
  Eigen::MatrixXd points_t = points.transpose();
  cv::eigen2cv(points_t, cv_points);
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
  Result result;
  for (int i = 0; i < n; i++)
  {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covar;
    double weight;
    cv::cv2eigen(means.row(i).t(), mean);
    cv::cv2eigen(covs[i], covar);
    weight = weights.at<double>(i);
    result.gmm.addGaussian(MultivariateGaussian(mean, covar), weight);
  }
  double ll = result.gmm.getLogLikelihood(points);
  size_t nb_parameters = getNbParameters(D, n, cov_mat_type);
  // TODO: labels
  result.scores[ScoreCriterion::LogLikelihood] = ll;
  result.scores[ScoreCriterion::AIC] = 2 * nb_parameters - 2 * ll;
  result.scores[ScoreCriterion::BIC] = nb_parameters * log(points.cols()) - 2 * ll;
  return result;
}
}  // namespace rhoban_random
