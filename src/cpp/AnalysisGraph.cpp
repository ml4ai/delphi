#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <random>
#include <sqlite3.h>
#include <utility>
#include <vector>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/range/irange.hpp>

#include "AnalysisGraph.hpp"

using boost::irange;

json load_json(string filename) {
  ifstream i(filename);
  json j;
  i >> j;
  return j;
}

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

map<string, vector<double>> construct_adjective_response_map() {
  sqlite3 *db;
  int rc = sqlite3_open(std::getenv("DELPHI_DB"), &db);
  if (!rc)
    COUT("Opened db successfully")
  else
    COUT("Could not open db")

  sqlite3_stmt *stmt;
  const char *query = "select * from gradableAdjectiveData";
  rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
  map<string, vector<double>> adjective_response_map;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    string adjective = std::string(
        reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2)));
    double response = sqlite3_column_double(stmt, 6);
    if (adjective_response_map.count(adjective) == 0) {
      cout << adjective << endl;
      adjective_response_map[adjective] = {response};
    } else {
      adjective_response_map[adjective].push_back(response);
    }
  }
  for (auto const &[k, v] : adjective_response_map) {
    cout << k << endl;
    for (auto resp : v) {
      cout << resp << endl;
    }
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return adjective_response_map;
}

/**
 * The AnalysisGraph class is the main model/interface for Delphi.
 */
class AnalysisGraph {
  DiGraph graph;

private:
  AnalysisGraph(DiGraph G) : graph(G){};

public:
  /**
   * A method to construct an AnalysisGraph object given a JSON-serialized list
   * of INDRA statements.
   *
   * @param filename The path to the file containing the JSON-serialized INDRA
   * statements.
   */
  static AnalysisGraph from_json_file(string filename) {
    auto j = load_json(filename);

    DiGraph G;
    std::unordered_map<string, int> node_int_map = {};
    int i = 0;
    for (auto stmt : j) {
      if (stmt["type"] == "Influence" and stmt["belief"] > 0.9) {
        auto subj = stmt["subj"]["concept"]["db_refs"]["UN"][0][0];
        auto obj = stmt["obj"]["concept"]["db_refs"]["UN"][0][0];
        if (!subj.is_null() and !obj.is_null()) {

          auto subj_string = subj.dump();
          auto obj_string = obj.dump();

          for (auto c : {subj_string, obj_string}) {
            if (node_int_map.count(c) == 0) {
              node_int_map[c] = i;
              i++;
              auto v = add_vertex(G);
              G[v].name = c;
            }
          }
          if (!edge(node_int_map[subj_string], node_int_map[obj_string], G)
                   .second) {
            auto edge = add_edge(node_int_map[subj_string],
                                 node_int_map[obj_string], G);
          }
        }
      }
    }
    return AnalysisGraph(G);
  }

  void construct_beta_pdfs() { COUT("Not implemented yet.") }
  auto print_nodes() {
    for_each(vertices(graph), [&](auto v) { cout << graph[v].name << endl; });
  }
  auto to_dot() {
    write_graphviz(cout, graph, make_label_writer(get(&Node::name, graph)));
  }
};
