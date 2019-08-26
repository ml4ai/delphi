#include "utils.hpp"
#include <fstream>
#include <boost/range/numeric.hpp>

using namespace std;

namespace utils {
  nlohmann::json load_json(string filename) {
    ifstream i(filename);
    nlohmann::json j = nlohmann::json::parse(i);
    return j;
  }

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

} // namespace utils
