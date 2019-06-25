#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sqlite3.h>
#include <utility>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>

#include "AnalysisGraph.hpp"

using std::cout, std::endl, std::unordered_map, std::pair, std::string,
    std::ifstream, std::stringstream, std::vector, std::map,
    boost::inner_product, boost::edge, boost::add_edge, boost::edge,
    boost::edges, boost::source, boost::target, boost::get, boost::graph_bundle,
    boost::make_label_writer, boost::write_graphviz, boost::range::for_each,
    boost::lambda::make_const;

using json = nlohmann::json;

template <class T, class U> bool hasKey(unordered_map<T, U> umap, T key) {
  return umap.count(key) == 0;
}

json load_json(string filename) {
  ifstream i(filename);
  json j;
  i >> j;
  return j;
}

size_t default_n_samples = 100;

unordered_map<string, vector<double>>
construct_adjective_response_map(size_t n_kernels = default_n_samples) {
  sqlite3 *db;
  int rc = sqlite3_open(std::getenv("DELPHI_DB"), &db);
  if (!rc)
    print("Opened db successfully");
  else
    print("Could not open db");

  sqlite3_stmt *stmt;
  const char *query = "select * from gradableAdjectiveData";
  rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
  unordered_map<string, vector<double>> adjective_response_map;

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    string adjective = std::string(
        reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2)));
    double response = sqlite3_column_double(stmt, 6);
    if (hasKey(adjective_response_map, adjective)) {
      adjective_response_map[adjective] = {response};
    } else {
      adjective_response_map[adjective].push_back(response);
    }
  }

  for (auto &[k, v] : adjective_response_map) {
    v = KDE(v).resample(n_kernels);
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
    auto json_data = load_json(filename);

    DiGraph G;
    std::unordered_map<string, int> nameMap = {};
    int i = 0;
    for (auto stmt : json_data) {
      if (stmt["type"] == "Influence" and stmt["belief"] > 0.9) {
        auto subj = stmt["subj"]["concept"]["db_refs"]["UN"][0][0];
        auto obj = stmt["obj"]["concept"]["db_refs"]["UN"][0][0];
        if (!subj.is_null() and !obj.is_null()) {

          auto subj_str = subj.dump();
          auto obj_str = obj.dump();

          // Add the nodes to the graph if they are not in it already
          for (auto c : {subj_str, obj_str}) {
            if (nameMap.count(c) == 0) {
              nameMap[c] = i;
              auto v = add_vertex(G);
              G[v].name = c;
              i++;
            }
          }

          // Add the edge to the graph if it is not in it already
          auto [e, exists] = add_edge(nameMap[subj_str], nameMap[obj_str], G);
          for (auto evidence : stmt["evidence"]) {
            auto annotations = evidence["annotations"];
            auto subj_adjectives = annotations["subj_adjectives"];
            auto obj_adjectives = annotations["obj_adjectives"];
            auto subj_adjective =
                (subj_adjectives.is_null() and subj_adjectives.size() > 0)
                    ? subj_adjectives[0]
                    : "";
            auto obj_adjective =
                (obj_adjectives.size() > 0) ? obj_adjectives[0] : "";
            auto subj_polarity = annotations["subj_polarity"];
            auto obj_polarity = annotations["obj_polarity"];
            G[e].causalFragments.push_back(CausalFragment{
                subj_adjective, obj_adjective, subj_polarity, obj_polarity});
          }
        }
      }
    }
    return AnalysisGraph(G);
  }

  void construct_beta_pdfs() {
    auto adjective_response_map = construct_adjective_response_map();
    vector<double> marginalized_responses;
    for (auto [adjective, responses] : adjective_response_map) {
      for (auto response:responses) {
        marginalized_responses.push_back(response);
      }
    }
    marginalized_responses = KDE(marginalized_responses).resample(default_n_samples);

    for_each(edges(graph), [&](auto e) {
      for (auto causalFragment : graph[e].causalFragments) {
        auto subj_adjective = causalFragment.subj_adjective;
        auto subj_polarity = causalFragment.subj_polarity;
        auto obj_adjective = causalFragment.obj_adjective;
        auto obj_polarity = causalFragment.obj_polarity;
        auto subject_responses = adjective_response_map[subj_adjective] | transformed(_1*subj_polarity);
        auto object_responses = adjective_response_map[obj_adjective] | transformed(_1*obj_polarity);
      }
    });
  }
  auto print_nodes() {
    for_each(vertices(graph), [&](auto v) { cout << graph[v].name << endl; });
  }
  auto to_dot() {
    write_graphviz(cout, graph, make_label_writer(get(&Node::name, graph)));
  }
};
