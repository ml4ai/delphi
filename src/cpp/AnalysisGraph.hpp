#ifndef ANALYSISGRAPH_H
#define ANALYSISGRAPH_H

#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/numeric.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/range/irange.hpp>

#include <nlohmann/json.hpp>
#define COUT(x) cout << x << endl;
#define CERR(x) cerr << x << endl;

namespace delphi {
  using std::cout,
        std::endl,
        std::unordered_map,
        std::pair,
        std::string,
        std::ifstream,
        std::stringstream,
        std::vector,
        std::map,
        boost::inner_product,
        boost::accumulate,
        boost::adaptors::transform,
        boost::adaptors::transformed,
        boost::adjacency_list,
        boost::edge,
        boost::add_edge,
        boost::vecS,
        boost::directedS,
        boost::edges,
        boost::source,
        boost::target,
        boost::irange,
        boost::get,
        boost::make_label_writer,
        boost::write_graphviz,
        boost::range::for_each,
        boost::lambda::_1,
        boost::lambda::_2,
        boost::lambda::make_const
  ;

  using json = nlohmann::json;
}

using namespace delphi;

double sqr(double x) { return x * x; }
double sum(vector<double> v) { return accumulate(v, 0.0); }
double mean(vector<double> v) { return sum(v) / v.size(); }

template <class T> T select_random_element(vector<T> v) {
  boost::random::mt19937 gen;
  boost::random::uniform_int_distribution<> dist(0, v.size() - 1);
  return v[dist(gen)];
}

double sample_from_normal(double mu = 0.0, double sd = 1.0) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{mu, sd};
  return d(gen);
}

// One-dimensional KDE
class KDE {
public:
  vector<double> dataset;
  double bw; // bandwidth

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

  double pdf(double x) {
    auto p = 0.0;
    auto N = dataset.size();
    for (auto elem : dataset) {
      auto x1 = exp(-sqr(x - elem) / (2 * sqr(bw)));
      x1 /= N * bw * sqrt(2 * M_PI);
      p += x1;
    }
    return p;
  }
  double logpdf(double x) { return log(pdf(x)); }
};

struct Node {
  string name;
};

struct CAGEdge {
  string name;
};

struct Model {
  string name;
};

typedef pair<int, int> Edge;
typedef adjacency_list<vecS, vecS, directedS, Node, CAGEdge, Model> DiGraph;

#endif
