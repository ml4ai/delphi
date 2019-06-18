#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>
#include <typeinfo>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graph_traits.hpp"
#include <boost/graph/graphviz.hpp>
#include <boost/range/algorithm/for_each.hpp>

#include <nlohmann/json.hpp>
#define print(x) cout << x << endl;


namespace delphi {
  using std::cout,
        std::endl,
        std::unordered_map,
        std::pair,
        std::string,
        std::ifstream;

  using boost::adjacency_list,
        boost::add_edge,
        boost::vecS,
        boost::directedS,
        boost::edges,
        boost::source,
        boost::target,
        boost::get,
        boost::make_label_writer,
        boost::write_graphviz,
        boost::range::for_each
  ;

  using json = nlohmann::json;
}

using namespace delphi;

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

json load_json(string filename) {
  ifstream i(filename);
  json j;
  i >> j;
  return j;
}

class AnalysisGraph {
public:
  DiGraph graph;
  AnalysisGraph(DiGraph G) : graph(G) {};
  static AnalysisGraph from_json_file(string filename) {
    auto j = load_json(filename);
    std::unordered_map<string, int> node_int_map = {};

    DiGraph G;
    int i = 0;
    for (auto stmt : j) {
      if (stmt["type"] == "Influence") {
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
          auto edge =
              add_edge(node_int_map[subj_string], node_int_map[obj_string], G);
        }
      }
    }
    return AnalysisGraph(G);
  }
};


int main(int argc, char *argv[]) {

  auto G = AnalysisGraph::from_json_file("indra_statements_format.json");
  write_graphviz(cout, G.graph, make_label_writer(get(&Node::name, G.graph)));

  return 0;
}
