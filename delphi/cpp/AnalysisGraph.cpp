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

json load_json(string filename) {
  ifstream i(filename);
  json j;
  i >> j;
  return j;
}

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
    std::unordered_map<string, int> nameMap = {};
    int i = 0;
    for (auto stmt : j) {
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
          if (!edge(nameMap[subj_str], nameMap[obj_str], G).second) {
            auto [e, exists] = add_edge(nameMap[subj_str], nameMap[obj_str], G);
            G[e].causalFragments.push_back(CausalFragment{"sa", "oa"});
          }
        }
      }
    }
    return AnalysisGraph(G);
  }

  void construct_beta_pdfs() {
    // auto adjective_response_map = construct_adjective_response_map();
    for_each(edges(graph), [&](auto e) {
      cout << graph[e].causalFragments[0].subj_adjective << endl;
    });
  }
  auto print_nodes() {
    for_each(vertices(graph), [&](auto v) { cout << graph[v].name << endl; });
  }
  auto to_dot() {
    write_graphviz(cout, graph, make_label_writer(get(&Node::name, graph)));
  }
};
