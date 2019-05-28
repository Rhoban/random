#include "rhoban_random/uniform_distribution.h"
#include <stdexcept>

#include <cmath>

namespace rhoban_random
{
UniformDistribution::UniformDistribution(double min_, double max_) : min(min_), max(max_)
{
  if (min >= max)
  {
    throw std::runtime_error("UniformDistribution : min is bigger or equal to max (forbidden).");
  }
}

double UniformDistribution::getLogLikelihood(double val) const
{
  if (val < min or val > max)
  {
    return std::numeric_limits<double>::lowest();
  }
  else
  {
    return -std::log(max - min);
  }
}

}  // namespace rhoban_random
