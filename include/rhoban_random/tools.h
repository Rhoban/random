#pragma once

#include <Eigen/Core>

#include <map>
#include <random>

namespace rhoban_random
{
std::default_random_engine getRandomEngine();
std::default_random_engine* newRandomEngine();

/// Return a vector of random_engines initialized with different seeds, if
/// engine is provided, then it is used to generate the random engines,
/// otherwise a new random_engine is created.
/// This function is particularly important to work with stochastic functions in
/// parallel
std::vector<std::default_random_engine> getRandomEngines(int nb_engines, std::default_random_engine* engine = NULL);

/// Return k distinct number chosen randomly from {0,1,...,n-1}
/// Create its own engine if no engine is provided
std::vector<size_t> getKDistinctFromN(size_t k, size_t n, std::default_random_engine* engine = NULL);

/// Similar to getKDistinctFromN, but if k >= n, then returns {0,1,...,n-1}
std::vector<size_t> getUpToKDistinctFromN(size_t k, size_t n, std::default_random_engine* engine = NULL);

/// Split the set {0, 1, ..., max_index} into multiple sets of the given size
/// throws a logic_error if sum(set_sizes) > max_index+1
/// Create its own engine if no engine is provided
std::vector<std::vector<size_t>> splitIndices(size_t max_index, const std::vector<size_t>& set_sizes,
                                              std::default_random_engine* engine = NULL);

/// Create its own engine if no engine is provided
std::vector<double> getUniformSamples(double min, double max, size_t nb_samples,
                                      std::default_random_engine* engine = NULL);

Eigen::VectorXd getUniformSample(const Eigen::MatrixXd& limits, std::default_random_engine* engine = NULL);

/// Create its own engine if no engine is provided
std::vector<Eigen::VectorXd> getUniformSamples(const Eigen::MatrixXd& limits, size_t nb_samples,
                                               std::default_random_engine* engine = NULL);

/// Each column of the result matrix is a different Sample
Eigen::MatrixXd getUniformSamplesMatrix(const Eigen::MatrixXd& limits, size_t nb_samples,
                                        std::default_random_engine* engine = NULL);

/// Given a vector of weight W, return a sample containing 'nb_samples' indices
/// with for each element: p(i) = W[i] / sum(W)
std::vector<int> sampleWeightedIndices(const std::vector<double>& weights, int nb_samples,
                                       std::default_random_engine* engine);

/// Return a map with for each index its number of occurences
/// cf sampleWeightedIndices
std::map<int, int> sampleWeightedIndicesMap(const std::vector<double>& weights, int nb_samples,
                                            std::default_random_engine* engine);
}  // namespace rhoban_random
