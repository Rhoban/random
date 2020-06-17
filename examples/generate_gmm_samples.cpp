#include <rhoban_random/gaussian_mixture_model.h>
#include <rhoban_random/tools.h>
#include <rhoban_utils/tables/double_table.h>

#include <tclap/CmdLine.h>

#include <iostream>

using namespace rhoban_random;

int main(int argc, char** argv)
{
  TCLAP::CmdLine cmd("Generate samples from a distribution and print them in a csv style", ' ', "0.0");
  TCLAP::ValueArg<std::string> path("p", "path", "The path to the json file describing the GMM", false, "gmm.json",
                                    "string", cmd);
  TCLAP::ValueArg<int> nb_samples("n", "nb-samples", "The number of samples to be generated", false, 10, "int", cmd);
  TCLAP::ValueArg<std::string> output("o", "output", "The output path for the samples file", false, "samples.csv",
                                      "string", cmd);

  try
  {
    cmd.parse(argc, argv);
  }
  catch (const TCLAP::ArgException& e)
  {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
  }

  MultivariateGaussian g1(Eigen::Vector2d(0.5, 1.0), Eigen::MatrixXd::Identity(2, 2));
  MultivariateGaussian g2(Eigen::Vector2d(-5.5, 5.0), 0.5 * Eigen::MatrixXd::Identity(2, 2));
  GaussianMixtureModel gmm({ g1, g2 }, { 0.6, 0.4 });
  if (path.isSet())
  {
    gmm.loadFile(path.getValue());
  }

  std::default_random_engine engine = rhoban_random::getRandomEngine();
  Eigen::MatrixXd samples = gmm.getSamples(nb_samples.getValue(), &engine);

  rhoban_utils::DoubleTable table(samples.transpose());

  table.writeFile(output.getValue());
}
