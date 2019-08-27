#pragma once

#include <random>
#include <boost/range/irange.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/range/numeric.hpp>
#include "utils.hpp"

using std::vector 
    , boost::accumulate
    , boost::adaptors::transformed
    , boost::irange
    , boost::lambda::_1
    , boost::lambda::_2
;

/**
 * Returns the square of a number.
 */
double sqr(double x) { return x * x; }

/**
 * Returns the sum of a vector of doubles.
 */
double sum(vector<double> v) { return accumulate(v, 0.0); }


/**
 * Returns the arithmetic mean of a vector of doubles.
 */
double mean(vector<double> v) { return sum(v) / v.size(); }

/**
 * Returns a randomly selected element of a vector.
 */
template <class T> T select_random_element(vector<T> v) {
  boost::random::mt19937 gen;
  boost::random::uniform_int_distribution<> dist(0, v.size() - 1);
  return v[dist(gen)];
}

double sample_from_normal(
    double mu = 0.0, /**< The mean of the distribution.*/
    double sd = 1.0  /**< The standard deviation of the distribution.*/
) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{mu, sd};
  return d(gen);
}

/**
 * The KDE class implements 1-D Gaussian kernel density estimators.
 */
class KDE {
public:
  KDE () {};
  vector<double> dataset;
  double bw; // bandwidth
  
  // TODO: Made this public just to initialize β.
  // Not sure this is the correct way to do it.
  double mu;

  KDE(vector<double> v) : dataset(v) {
    // Compute the bandwidth using Silverman's rule
    auto mu = mean(v);
    auto X = v | transformed(_1 - mu);

    // Compute standard deviation of the sample.
    auto N = v.size();
    auto stdev = sqrt(inner_product(X, X, 0.0) / (N - 1));
    bw = pow(4 * pow(stdev, 5) / (3 * N), 1 / 5);
  }

  auto resample(int n_samples) {
    vector<double> samples;
    for (int i : irange(0, n_samples)) {
      auto element = select_random_element(dataset);
      samples.push_back(sample_from_normal(element, bw));
    }
    return samples;
  }

  auto pdf(double x) {
    auto p = 0.0;
    auto N = dataset.size();
    for (auto elem : dataset) {
      auto x1 = exp(-sqr(x - elem) / (2 * sqr(bw)));
      x1 /= N * bw * sqrt(2 * M_PI);
      p += x1;
    }
    return p;
  }

  auto pdf(vector<double> v) {
    vector<double> values;
    for (auto elem : v) {
      values.push_back(pdf(elem));
    }
    return values;
  }

  auto logpdf(double x) { return log(pdf(x)); }
};
