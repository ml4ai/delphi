#include "utils.hpp"
#include <cmath>
#include <fstream>
#include <boost/range/numeric.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/median.hpp>

using namespace std;

namespace delphi::utils {

/**
 * Returns the square of a number.
 */
double sqr(double x) { return x * x; }

/**
 * Returns the sum of a vector of doubles.
 */
double sum(const std::vector<double> &v) { return boost::accumulate(v, 0.0); }

/**
 * Returns the arithmetic mean of a vector of doubles.
 * Updated based on:
 * https://codereview.stackexchange.com/questions/185450/compute-mean-variance-and-standard-deviation-of-csv-number-file
 */
double mean(const std::vector<double> &v) {
    if (v.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return sum(v) / v.size();
}

/**
 * Returns the sample standard deviation of a vector of doubles.
 * Based on:
 * https://codereview.stackexchange.com/questions/185450/compute-mean-variance-and-standard-deviation-of-csv-number-file
 */
double standard_deviation(const double mean, const std::vector<double>& v)
{
    if (v.size() <= 1u)
        return std::numeric_limits<double>::quiet_NaN();

    auto const add_square = [mean](double sum, int i) {
        auto d = i - mean;
        return sum + d*d;
    };
    double total = std::accumulate(v.begin(), v.end(), 0.0, add_square);
    return sqrt(total / (v.size() - 1));
}

/**
 * Returns the median of a vector of doubles.
 */
double median(const std::vector<double> &xs) {
    if (xs.size() > 100) {
        using namespace boost::accumulators;
        accumulator_set<double, features<tag::median>> acc;
        //  accumulator_set<double,
        //      features<tag::median(with_p_square_cumulative_distribution) >>
        //      acc ( p_square_cumulative_distribution_num_cells = xs.size() );

        for (auto x : xs) {
          acc(x);
        }

        return boost::accumulators::median(acc);
    } else {
        vector<double> x_copy(xs);
        sort(x_copy.begin(), x_copy.end());
        int num_els = x_copy.size();
        int mid = num_els / 2;
        if (num_els % 2 == 0) {
            return (x_copy[mid - 1] +  x_copy[mid]) / 2;
        }
        else {
            return x_copy[mid];
        }
    }
}

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
