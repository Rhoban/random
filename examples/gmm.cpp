#include <iostream>
#include "rhoban_random/gaussian_mixture_model.h"
#include "rhoban_random/multivariate_gaussian.h"

using namespace rhoban_random;

int main()
{
  std::cout << "GMM" << std::endl;

  MultivariateGaussian mv;
  mv.loadFile("gmm.json");

  /*
    Eigen::VectorXd mean(2);
    mean << 1.0, 2.0;
    Eigen::MatrixXd covar(2, 2);
    covar << 1.0, 0.5, 0.5, 1.0;
    MultivariateGaussian mv1(mean, covar);
  */
}