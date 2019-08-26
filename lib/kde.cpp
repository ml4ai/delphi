#include <random>
#include <boost/range/irange.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/range/numeric.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include "kde.hpp"

using std::vector;
using boost::irange;
using boost::lambda::_1;
using boost::lambda::_2;
using std::mt19937;
using std::normal_distribution;


double sample_from_normal(
    double mu = 0.0, /**< The mean of the distribution.*/
    double sd = 1.0  /**< The standard deviation of the distribution.*/
) {
  mt19937 gen = RNG::rng()->get_RNG();
  normal_distribution<> d{mu, sd};
  return d(gen);
}

KDE::KDE(std::vector<double> v) : dataset(v) {
    using boost::adaptors::transformed;
    using boost::lambda::_1;
    using utils::mean;

    // Compute the bandwidth using Silverman's rule
    mu = mean(v);
    auto X = v | transformed(_1 - mu);

    // Compute standard deviation of the sample.
    size_t N = v.size();
    double stdev = sqrt(inner_product(X, X, 0.0) / (N - 1));
    bw = pow(4 * pow(stdev, 5) / (3 * N), 1 / 5);
  }

vector<double> KDE::resample(int n_samples) {
  vector<double> samples;
  for (int i : irange(0, n_samples)) {
    double element = select_random_element(dataset);
    samples.push_back(sample_from_normal(element, bw));
  }
  return samples;
}

double KDE::pdf(double x) {
  using utils::sqr;
  double p = 0.0;
  size_t N = dataset.size();
  for (double elem : dataset) {
    double x1 = exp(-sqr(x - elem) / (2 * sqr(bw)));
    x1 /= N * bw * sqrt(2 * M_PI);
    p += x1;
  }
  return p;
}

vector<double> KDE::pdf(vector<double> v) {
  vector<double> values;
  for (double elem : v) {
    values.push_back(pdf(elem));
  }
  return values;
}

double KDE::logpdf(double x) { return log(pdf(x)); }

