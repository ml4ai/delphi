#pragma once

#include "rng.hpp"
#include "utils.hpp"
#include <boost/range/numeric.hpp>
#include <random>
#include "dbg.h"

/**
 * Returns a randomly selected element of a vector.
 */
template <class T> T select_random_element(std::vector<T> v, std::mt19937& gen, std::uniform_real_distribution<double>& uni_dist) {
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
    //uniform_int_distribution<> dist(0, v.size() - 1);
    double rand_val = uni_dist(gen);
    int idx = trunc(rand_val * v.size());
    //int idx = trunc(uni_dist(gen) * v.size());
    idx = idx == v.size()? idx-- : idx;
    dbg(rand_val);
    dbg(rand_val * v.size());
    dbg(idx);
    element = v[idx];
    //element = v[dist(gen)];
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

  std::vector<double> resample(int n_samples, std::mt19937& rng,
                        std::uniform_real_distribution<double>& uni_dist,
                        std::normal_distribution<double>& norm_dist);
  std::vector<double> resample(int n_samples);
  double pdf(double x);
  std::vector<double> pdf(std::vector<double> v);
  double logpdf(double x);
};
