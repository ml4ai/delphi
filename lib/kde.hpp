#pragma once

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/range/numeric.hpp>
#include "rng.hpp"
#include "utils.hpp"

/**
 * Returns a randomly selected element of a vector.
 */
template <class T> T select_random_element(std::vector<T> v) {
  using namespace std;
  mt19937 gen = RNG::rng()->get_RNG();
  T element;
  if (v.size() == 0) {
    throw "Vector is empty, so we cannot select a random element from it. (function: select_random_element)\n";
  }
  else if (v.size() == 1){
    element = v[0];
  }
  else {
    boost::random::uniform_int_distribution<> dist(0, v.size() - 1);
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
  KDE () {};
  std::vector<double> dataset;
  double bw; // bandwidth
  KDE(std::vector<double>);
  
  // TODO: Made this public just to initialize β.
  // Not sure this is the correct way to do it.
  double mu;


  std::vector<double> resample(int n_samples);
  double pdf(double x);
  std::vector<double> pdf(std::vector<double> v);
  double logpdf(double x);
};
