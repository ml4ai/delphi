#include "kde.hpp"
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/numeric.hpp>
#include <random>

using namespace std;
using boost::irange;
using boost::adaptors::transformed;
using boost::lambda::_1;
using namespace delphi::utils;

double sample_from_normal(
    std::mt19937 gen,
    double mu = 0.0, /**< The mean of the distribution.*/
    double sd = 1.0  /**< The standard deviation of the distribution.*/
) {
  normal_distribution<> d{mu, sd};
  return d(gen);
}

KDE::KDE(std::vector<double> v) : dataset(v) {

  // Compute the bandwidth using Silverman's rule
  mu = mean(v);
  auto X = v | transformed(_1 - mu);

  // Compute standard deviation of the sample.
  size_t N = v.size();
  double stdev = sqrt(inner_product(X, X, 0.0) / (N - 1));
  bw = pow(4 * pow(stdev, 5) / (3 * N), 1 / 5);
}

vector<double> KDE::resample(int n_samples, std::mt19937 gen,
                        uniform_real_distribution<double>& uni_dist,
                        normal_distribution<double>& norm_dist) {
  vector<double> samples;
  //uniform_int_distribution<int> uni_dist(0, dataset.size() - 1);
  //uniform_real_distribution<double> uni_dist(0, 1);
  //normal_distribution<double> norm_dist{0.0, 1.0};

  for (int i : irange(0, n_samples)) {
    double element = select_random_element(dataset, gen, uni_dist);
    //samples.push_back(sample_from_normal(gen, element, bw));
    samples.push_back(element + bw * norm_dist(gen));
  }
  return samples;
}

double KDE::pdf(double x) {
  double p = 0.0;
  size_t N = this->dataset.size();
  for (double elem : this->dataset) {
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
