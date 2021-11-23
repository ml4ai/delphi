#include "KDE.hpp"
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/numeric.hpp>
#include <random>
#include <math.h>
//#include "dbg.h"

using namespace std;
using namespace delphi::utils;
using boost::irange, boost::adaptors::transformed, boost::lambda::_1;

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

KDE::KDE(vector<double> thetas, int n_bins)  : n_bins(n_bins) {
  this->dataset = thetas;
  double small_count = 0.00001; // To avoid log(0)
  this->log_prior_hist = vector<double>(n_bins, small_count);
  this->delta_theta = M_PI / n_bins;
//  this->delta_theta = 2.0 / n_bins;

//  int bin_lo = n_bins - 1;
//  int bin_hi = 0;

//  dbg(delta_theta);
//  dbg(log_prior_hist);

  for (double theta : thetas) {
    theta = theta < 0 ? M_PI + theta : theta;
//    theta = theta < 0 ? 2 + theta : theta;

    int bin = this->theta_to_bin(theta);
//    bin_lo = bin < bin_lo ? bin : bin_lo;
//    bin_hi = bin > bin_hi ? bin : bin_hi;

//    dbg(bin);
    this->log_prior_hist[bin] += 1;
  }
//  dbg(log_prior_hist);
//  dbg(bin_lo);
//  dbg(bin_hi);

//  if (bin_lo != bin_hi && bin_lo != (bin_hi + 1) % n_bins)

  double n_points = thetas.size() + small_count * n_bins;

  for (double & count : this->log_prior_hist) {
    count /= n_points;
//    dbg(count);
    count = log(count);
  }
//  dbg(log_prior_hist);
//  this->dataset = log_prior_hist;
}

void KDE::set_num_bins(int n_bins) {
  this->n_bins = n_bins;
  this->delta_theta = M_PI / n_bins;
}

int KDE::theta_to_bin(double theta) {
    return floor(theta / this->delta_theta);
}


vector<double> KDE::resample(int n_samples,
                             std::mt19937& gen,
                             uniform_real_distribution<double>& uni_dist,
                             normal_distribution<double>& norm_dist) {
  vector<double> samples(n_samples);

  for (int i : irange(0, n_samples)) {
    double element = select_random_element(dataset, gen, uni_dist);

    // Transform the sampled values using a Gaussian distribution
    // ~ ( sampled value, bw)
    // We sample from a standard Gaussian and transform that sample
    // to the desired Gaussian distribution by
    // μ + σ * standard Gaussian sample
    samples[i] = element + bw * norm_dist(gen);
  }

  return samples;
}

double KDE::pdf(double x) {
//  dbg("Should not be called!");
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
