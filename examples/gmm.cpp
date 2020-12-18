#include <iostream>
#include "rhoban_random/gaussian_mixture_model.h"
#include "rhoban_random/multivariate_gaussian.h"

using namespace rhoban_random;

MultivariateGaussian getMixture(double m1, double m2, double cv11, double cv12, double cv21, double cv22)
{
  Eigen::VectorXd mean(2);
  mean << m1, m2;
  Eigen::MatrixXd covar(2, 2);
  covar << cv11, cv12, cv21, cv22;
  MultivariateGaussian g(mean, covar);

  return g;
}

int main()
{
  GaussianMixtureModel gmm;
  gmm.addGaussian(getMixture(1.0, 0.0, 1.0, 0.8, 0.5, 1.5), 0.5);
  gmm.addGaussian(getMixture(10.0, 1.0, 0.5, 0.01, 0.01, 0.1), 0.5);
  gmm.saveFile("gmm.json");

  std::default_random_engine e;
  for (int k = 0; k < 1000; k++)
  {
    auto sample = gmm.getSample(&e);
    std::cout << sample[0] << " " << sample[1] << std::endl;
    ;
  }
}