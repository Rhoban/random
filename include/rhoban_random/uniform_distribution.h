#pragma once

namespace rhoban_random
{
class UniformDistribution
{
public:
  UniformDistribution(double min, double max);

  /// Return the loglikelihood of given value according to distribution
  double getLogLikelihood(double value) const;

private:
  double min;
  double max;
};

}  // namespace rhoban_random
