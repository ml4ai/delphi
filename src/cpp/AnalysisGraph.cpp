#include <pybind11/pybind11.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>
#include <typeinfo>
#include <sqlite3.h>
#include <numeric>
#include <cmath>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>


#include "AnalysisGraph.hpp"

json load_json(string filename) {
  ifstream i(filename);
  json j;
  i >> j;
  return j;
}


// One-dimensional KDE
struct KDE {
  vector<double> dataset;
};

double sum(vector<double> v){
  return accumulate(v, 0.0);
}

double sqr(double x) {
  return x*x;
}

double mean (vector<double> v) {
  return accumulate(v, 0.0)/v.size();
}

auto construct_adjective_response_map() {
  sqlite3 *db;
  int rc = sqlite3_open(std::getenv("DELPHI_DB"), &db);
  if (!rc) COUT("Opened db successfully")
  else COUT("Could not open db")

  sqlite3_stmt *stmt;
  const char* query = "select * from gradableAdjectiveData";
  rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
  map<string, vector<double> > adjective_response_map;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      string adjective = std::string(reinterpret_cast<const char*>(
        sqlite3_column_text(stmt, 2)
      ));
      double response = sqlite3_column_double(stmt, 6);
      if (adjective_response_map.count(adjective) == 0) {
        cout << adjective << endl;
        adjective_response_map[adjective] = {response};
      }
      else {adjective_response_map[adjective].push_back(response);}
  }
  for (auto const& [k, v] : adjective_response_map) {
    cout << k << endl;
    for (auto resp : v) {
      cout << resp << endl;
    }
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return adjective_response_map;
}

class AnalysisGraph {
  DiGraph graph;
private:
  AnalysisGraph(DiGraph G) : graph(G) {};
public:
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
          if (!edge(node_int_map[subj_string], node_int_map[obj_string], G).second){
            auto edge =
              add_edge(node_int_map[subj_string], node_int_map[obj_string], G);
          }
        }
      }
    }
    return AnalysisGraph(G);
  }

  auto construct_beta_pdfs(){
    auto adjective_response_map = construct_adjective_response_map();
    for (auto item : adjective_response_map) {

      // Compute the bandwidth using Silverman's rule

      auto v = item.second;
      auto mu = mean(v);
      auto X = v | transformed(_1 - mu);

      // Compute standard deviation of the sample.
      auto N = v.size(); 
      auto stdev = sqrt(inner_product(X, X, 0.0)/(N-1));
      auto bw = pow(4*pow(stdev, 5)/(3*N), 1/5);
      // 1-D Gaussian KDE (Bishop eq. 2.250)
      auto pdf = [&] (auto x) {
        auto p = 0.0;
        for (auto elem : v) {
          auto x1 = exp(-sqr(x-elem)/(2*sqr(bw)));
          x1 /= N*bw*sqrt(2*M_PI);
          p+= x1;
        }
        return p;
      };
      return pdf;
    }
  }
  auto print_nodes(){
    for_each(vertices(graph), [&] (auto v) {cout << graph[v].name << endl;});
  }
  auto to_dot(){
    write_graphviz(cout, graph, make_label_writer(get(&Node::name, graph)));
  }
};
