#include <rhoban_random/gaussian_mixture_model.h>
#include <rhoban_random/tools.h>

#include <iostream>

using namespace rhoban_random;

int main()
{
  // Creating a simple example of 3 gaussians of dimension 2
  MultivariateGaussian g1(Eigen::Vector2d(0.5, 1.0), Eigen::MatrixXd::Identity(2, 2));
  MultivariateGaussian g2(Eigen::Vector2d(-15.5, 100.0), 3 * Eigen::MatrixXd::Identity(2, 2));
  MultivariateGaussian g3(Eigen::Vector2d(20.5, -10.0), 12 * Eigen::MatrixXd::Identity(2, 2));
  GaussianMixtureModel gmm({ g1, g2, g3 }, { 0.3, 0.5, 0.2 });
  // Acquiring samples
  std::default_random_engine engine = getRandomEngine();
  int nb_samples = 100;
  Eigen::MatrixXd samples = gmm.getSamples(nb_samples, &engine);
  // std::cout << "samples:" << std::endl << samples.transpose() << std::endl;
  // Fitting EM
  GaussianMixtureModel fitted_gmm =
      GaussianMixtureModel::EM(3, samples, GaussianMixtureModel::EMCovMatType::COV_MAT_DIAGONAL, nullptr, 1000);
}
