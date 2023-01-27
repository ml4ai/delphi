#pragma once

#include "rng.hpp"
#include "utils.hpp"
#include <boost/range/numeric.hpp>
#include <random>

/**
 * Returns a randomly selected element of a vector.
 */
template <class T>
T select_random_element(std::vector<T> v,
                        std::mt19937& gen,
                        std::uniform_real_distribution<double>& uni_dist) {
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
    // Sample a double between [0, 1]
    double rand_val = uni_dist(gen);

    // Convert random double between [0, 1] into
    // an integer between [0, v.size() - 1
    unsigned long int idx = trunc(rand_val * v.size());
    if(idx == v.size()) {
      idx--;
    }

    element = v[idx];
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
  std::vector<double> dataset = {};
  std::vector<double> log_prior_hist = {};
  double delta_theta = 1;
  int n_bins = 1;
  double bw; // bandwidth
  KDE(std::vector<double>);
  KDE(std::vector<double>, int n_bins);
  void set_num_bins(int n_bins);
  int theta_to_bin(double theta);
  double bin_to_theta(int bin);

  // TODO: Made this public just to initialize β.
  // Not sure this is the correct way to do it.
  double mu = 0;
  double most_probable_theta = 0;

  std::vector<double> resample(int n_samples,
                               std::mt19937& rng,
                               std::uniform_real_distribution<double>& uni_dist,
                               std::normal_distribution<double>& norm_dist);
  std::vector<double> resample(int n_samples);
  double pdf(double x);
  std::vector<double> pdf(std::vector<double> v);
  double logpdf(double x);
};
