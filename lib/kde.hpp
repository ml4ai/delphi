#pragma once

#include "rng.hpp"
#include "utils.hpp"
#include <boost/range/numeric.hpp>
#include <random>

/**
 * Returns a randomly selected element of a vector.
 */
template <class T> T select_random_element(std::vector<T> v, std::mt19937 gen) {
  using namespace std;
  T element;
  if (v.size() == 0) {
    throw runtime_error(
        "Vector is empty, so we cannot select a random element from it. "
        "(function: select_random_element)\n");
  }
  else if (v.size() == 1) {
    element = v[0];
  }
  else {
    uniform_int_distribution<> dist(0, v.size() - 1);
    element = v[dist(gen)];
  }
  return element;
}

double sample_from_normal(double mu, double sd);

/**
 * The KDE class implements 1-D Gaussian kernel density estimators.
 */
class KDE {
  public:
  KDE(){};
  std::vector<double> dataset;
  double bw; // bandwidth
  KDE(std::vector<double>);

  // TODO: Made this public just to initialize β.
  // Not sure this is the correct way to do it.
  double mu;

  std::vector<double> resample(int n_samples, std::mt19937 rng);
  std::vector<double> resample(int n_samples);
  double pdf(double x);
  std::vector<double> pdf(std::vector<double> v);
  double logpdf(double x);
};
