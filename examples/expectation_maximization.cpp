#include <rhoban_random/expectation_maximization.h>

#include <rhoban_utils/tables/double_table.h>

#include <tclap/CmdLine.h>
#include <iostream>

using namespace rhoban_random;

int main(int argc, char** argv)
{
  TCLAP::CmdLine cmd("Generate samples from a distribution and print them in a csv style", ' ', "0.0");
  TCLAP::ValueArg<std::string> input("i", "input", "The path to the csv file containing the samples", true,
                                     "samples.csv", "string", cmd);
  // TODO add json option for config of ExpectationMaximization
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
  ExpectationMaximization em;
  em.analyze(samples);
  // Score results
  em.getScoresTable().dump("results.csv");
  em.getBestResult(ExpectationMaximization::ScoreCriterion::AIC).gmm.saveFile("best_aic.json");
  em.getBestResult(ExpectationMaximization::ScoreCriterion::BIC).gmm.saveFile("best_bic.json");
}
