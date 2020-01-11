#include "utils.hpp"
#include <cmath>
#include <fstream>
#include <boost/range/numeric.hpp>

using namespace std;

namespace delphi::utils {

/**
 * Returns the square of a number.
 */
double sqr(double x) { return x * x; }

/**
 * Returns the sum of a vector of doubles.
 */
double sum(std::vector<double> v) { return boost::accumulate(v, 0.0); }

/**
 * Returns the arithmetic mean of a vector of doubles.
 */
double mean(std::vector<double> v) { return sum(v) / v.size(); }

double log_normpdf(double x, double mean, double sd) {
  double var = pow(sd, 2);
  double log_denom = -0.5 * log(2 * M_PI) - log(sd);
  double log_nume = pow(x - mean, 2) / (2 * var);

  return log_denom - log_nume;
}

nlohmann::json load_json(string filename) {
  ifstream i(filename);
  nlohmann::json j = nlohmann::json::parse(i);
  return j;
}

} // namespace delphi::utils
