#pragma once

#include <random>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/range/numeric.hpp>
#include "rng.hpp"

/**
 * Returns the square of a number.
 */
double sqr(double x);

/**
 * Returns the sum of a vector of doubles.
 */
double sum(std::vector<double> v);


/**
 * Returns the arithmetic mean of a vector of doubles.
 */
double mean(std::vector<double> v);

/**
 * Returns a randomly selected element of a vector.
 */
template <class T> T select_random_element(std::vector<T> v) {
  std::mt19937 gen = RNG::rng()->get_RNG();
  boost::random::uniform_int_distribution<> dist(0, v.size() - 1);
  return v[dist(gen)];
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
  
  // TODO: Made this public just to initialize β.
  // Not sure this is the correct way to do it.
  double mu;

  KDE(std::vector<double> v) : dataset(v) {
    using boost::adaptors::transformed;
    using boost::lambda::_1;

    // Compute the bandwidth using Silverman's rule
    mu = mean(v);
    auto X = v | transformed(_1 - mu);

    // Compute standard deviation of the sample.
    size_t N = v.size();
    double stdev = sqrt(inner_product(X, X, 0.0) / (N - 1));
    bw = pow(4 * pow(stdev, 5) / (3 * N), 1 / 5);
  }

  std::vector<double> resample(int n_samples);
  double pdf(double x);
  std::vector<double> pdf(std::vector<double> v);
  double logpdf(double x);
};
