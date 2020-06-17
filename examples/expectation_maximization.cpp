#include <rhoban_random/gaussian_mixture_model.h>
#include <rhoban_random/tools.h>

#include <rhoban_utils/tables/double_table.h>

#include <tclap/CmdLine.h>
#include <iostream>

using namespace rhoban_random;

int main(int argc, char** argv)
{
  TCLAP::CmdLine cmd("Generate samples from a distribution and print them in a csv style", ' ', "0.0");
  TCLAP::ValueArg<std::string> input("i", "input", "The path to the csv file containing the samples", true,
                                     "samples.csv", "string", cmd);
  try
  {
    cmd.parse(argc, argv);
  }
  catch (const TCLAP::ArgException& e)
  {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
  }
  rhoban_utils::DoubleTable data = rhoban_utils::DoubleTable::buildFromFile(input.getValue());
  Eigen::MatrixXd samples = data.getData().transpose();
  // Fitting EM
  // TODO: test all types
  std::cout << "nb_clusters,ll,aic,bic" << std::endl;
  for (int n = 1; n < 16; n++)
  {
    GaussianMixtureModel fitted_gmm =
        GaussianMixtureModel::EM(n, samples, GaussianMixtureModel::EMCovMatType::COV_MAT_SPHERICAL, nullptr, 10, 0.01);
    double ll = fitted_gmm.getLogLikelihood(samples);
    int nb_parameters = 4 * n - 1;
    double aic = 2 * nb_parameters - 2 * ll;
    double bic = nb_parameters * log(samples.cols()) - 2 * ll;
    std::cout << n << "," << ll << "," << aic << "," << bic << std::endl;
  }
}
