#include <gtest/gtest.h>
#include <rhoban_random/gaussian_mixture_model.h>

#define EPSILON std::pow(10, -6)

using namespace rhoban_random;

TEST(constructor, empty)
{
  GaussianMixtureModel gmm;
  EXPECT_EQ(0, gmm.size());
  EXPECT_EQ(0, gmm.dimension());
}

TEST(constructor, singleElement)
{
  MultivariateGaussian g(Eigen::Vector2d(0.5, 1.0), Eigen::MatrixXd::Identity(2, 2));
  GaussianMixtureModel gmm({ g }, { 1.0 });
  EXPECT_EQ(1, gmm.size());
  EXPECT_EQ(2, gmm.dimension());
}

TEST(constructor, multipleElements)
{
  MultivariateGaussian g1(Eigen::Vector2d(0.5, 1.0), Eigen::MatrixXd::Identity(2, 2));
  MultivariateGaussian g2(Eigen::Vector2d(1.5, 1.0), Eigen::MatrixXd::Identity(2, 2));
  GaussianMixtureModel gmm({ g1, g2, g1 }, { 1.0, 0.5, 1.0 });
  EXPECT_EQ(3, gmm.size());
  EXPECT_EQ(2, gmm.dimension());
}

TEST(constructor, negativeWeight)
{
  MultivariateGaussian g(Eigen::Vector2d(0.5, 1.0), Eigen::MatrixXd::Identity(2, 2));
  try
  {
    std::vector<double> weights = { -1.0 };
    GaussianMixtureModel gmm({ g }, weights);
    FAIL() << "Using a negativeWeight for a gaussian should throw an error" << std::endl;
  }
  catch (const std::logic_error& err)
  {
  }
}

TEST(likelihood, uniqueGaussian)
{
  MultivariateGaussian g(Eigen::Vector2d(0.5, 1.0), Eigen::MatrixXd::Identity(2, 2));
  GaussianMixtureModel gmm({ g }, { 0.3 });
  Eigen::Vector2d p(0, 0);
  EXPECT_NEAR(g.getLikelihood(p), gmm.getLikelihood(p), EPSILON);
  EXPECT_NEAR(g.getLogLikelihood(p), gmm.getLogLikelihood(p), EPSILON);
}

TEST(likelihood, multipleGaussians)
{
  MultivariateGaussian g1(Eigen::Vector2d(0.5, 1.0), Eigen::MatrixXd::Identity(2, 2));
  MultivariateGaussian g2(Eigen::Vector2d(-0.5, 1.0), 3 * Eigen::MatrixXd::Identity(2, 2));
  MultivariateGaussian g3(Eigen::Vector2d(0.5, -1.0), 12 * Eigen::MatrixXd::Identity(2, 2));
  GaussianMixtureModel gmm({ g1, g2, g3 }, { 0.3, 0.5, 0.2 });
  Eigen::Vector2d p(0, 0);
  double expected_L = 0.3 * g1.getLikelihood(p) + 0.5 * g2.getLikelihood(p) + 0.2 * g3.getLikelihood(p);
  double expected_LL = log(expected_L);
  EXPECT_NEAR(expected_L, gmm.getLikelihood(p), EPSILON);
  EXPECT_NEAR(expected_LL, gmm.getLogLikelihood(p), EPSILON);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
