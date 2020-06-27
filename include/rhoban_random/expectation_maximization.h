#pragma once

#include <rhoban_random/gaussian_mixture_model.h>

#include <rhoban_utils/tables/string_table.h>

namespace rhoban_random
{
/// This class implements the Expectation-Maximization algorithm to find a Gaussian Mixture Model
///
/// It can be used to test multiple configurations and keep the best one according to an information criteria such as
/// BIC or AIC
///
/// It has the following properties
/// - Keep memory of multiple configurations
/// - Allows to retrieve for different configurations: BIC score or AIC score
class ExpectationMaximization : public rhoban_utils::JsonSerializable
{
public:
  /// Constraints on covariance Matrix for EM algorithm
  /// TODO: missing covMatType with respect to usually used: Tied (one matrix for all of them)
  enum CovMatType
  {
    /// Cov Matrix is a scaled Identity matrix ($\alpha I$ with $\alpha the only free parameter)
    Spherical,
    /// Cov Matrix is a diagonal matrix with one free parameter per dimension
    Diagonal,
    /// Cov Matrix is symmetric positively defined
    Generic
  };

  enum ScoreCriterion
  {
    /// Bayesian Information Criterion
    BIC,
    /// Akaike Information Criterion
    AIC,
    /// LogLikelihood of the points according to the GMM
    LogLikelihood
  };

  /// A configuration for the EM algorithm:
  /// - First element is number of clusters
  /// - Second element is the type of covariance matrix used
  typedef std::pair<size_t, CovMatType> Configuration;

  struct Result
  {
    /// The GaussianMixtureModel after EM
    GaussianMixtureModel gmm;
    /// The labels
    Eigen::VectorXi labels;
    /// The scores according to various metrics
    std::map<ScoreCriterion, double> scores;
  };

  ExpectationMaximization();

  void setMinClusters(int new_min);
  void setMaxClusters(int new_max);
  void setAllowedCovMatTypes(const std::set<CovMatType>& allowed_types);

  /// Analyze the provided data with all the allowed configurations:
  /// - Results can be retrieved afterwards using getBestResult, getResult, getBestConfiguration
  void analyze(const Eigen::MatrixXd& data);

  /// Throws a out_of_range if no analysis has been performed
  Configuration getBestConfiguration(ScoreCriterion criterion) const;
  /// Throws a out_of_range if no analysis has been performed
  const Result& getBestResult(ScoreCriterion criterion) const;
  /// Throws a out_of_range if analysis has not been performed for configuration c
  const Result& getResult(const Configuration& c) const;

  rhoban_utils::StringTable getScoresTable() const;

  static std::string toString(rhoban_random::ExpectationMaximization::CovMatType type);

  std::string getClassName() const override;
  void fromJson(const Json::Value& value, const std::string& dir_name) override;
  Json::Value toJson() const override;

private:
  /// The points for which a GaussianMixtureModel has to be found
  Eigen::MatrixXd points;

  /// The result of the EM process by configuration
  std::map<Configuration, Result> entries;

  /// Minimal number of clusters
  int min_clusters;

  /// Maximal number of clusters
  int max_clusters;

  /// The maximal number of steps for the EM algorithm
  int max_iterations;

  /// The minimal score improvement for the EM algorithm
  double epsilon;

  std::set<CovMatType> cov_mat_types_allowed;

  /// Clear history of entries and update the points to be used for analysis
  void setPoints(const Eigen::MatrixXd& new_points);

  /// Return the number of parameters associated to the given configuration
  /// dimensions: The number of dimensions for the mixture
  /// nb_clusters: The number of clusters
  static size_t getNbParameters(size_t dimensions, size_t nb_clusters, CovMatType cov_mat_type);

  /// Uses the ExpectationMaximization algorithm to obtain a gaussian mixture model based on the provided points
  /// - n: number of clusters
  /// - inputs: each column is a different point
  /// - labels: if not null, is filled with the id of the gaussian attributed to each element
  /// - max_iterations: the maximal number of iteration for the EM algorithm
  /// - epsilon: if improvement of logLikelihood goes below this value, stop iterations
  Result run(int n, CovMatType cov_mat_type);
};

}  // namespace rhoban_random
