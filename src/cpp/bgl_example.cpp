#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graph_traits.hpp"

#include <nlohmann/json.hpp>

#define print(X) cout << X << endl;

using namespace boost;
using namespace std;
using json = nlohmann::json;

struct Node {
  string name;
};

struct CAGEdge {
  string name;
};

struct Model {
  string name;
};

// An edge is just a connection between two vertitices. Our verticies above
// are an enum, and are just used as integers, so our edges just become
// a pair<int, int>
typedef pair<int, int> Edge;

// create an directed- graph type, using vectors as the underlying containers
// and an adjacency_list as the basic representation
typedef adjacency_list<vecS, vecS, directedS, Node, CAGEdge, Model>
    AnalysisGraph;

json get_json_from_file(const char *filename) {
  ifstream i(filename);
  json j;
  i >> j;
  return j;
}

int main(int argc, char *argv[]) {

  auto j = get_json_from_file("indra_statements_format.json");
  for (auto stmt : j) {
    if (stmt["type"] == "Influence") {
      print(stmt)
    }
  }

  // Example uses an array, but we can easily use another container type
  // to hold our edges.
  // vector<Edge> edgeVec;
  // edgeVec.push_back(Edge(1, 2));
  // edgeVec.push_back(Edge(1, 4));
  // edgeVec.push_back(Edge(3, 4));
  // edgeVec.push_back(Edge(4, 3));
  // edgeVec.push_back(Edge(3, 5));
  // edgeVec.push_back(Edge(2, 4));
  // edgeVec.push_back(Edge(4, 5));

  //// Now we can initialize our graph using iterators from our above vector
  // AnalysisGraph g(edgeVec.begin(), edgeVec.end(), 6);

  // cout << num_edges(g) << "\n";

  //// Ok, we want to see that all our edges are now contained in the graph
  // typedef graph_traits<AnalysisGraph>::edge_iterator edge_iterator;

  //// Tried to make this section more clear, instead of using tie, keeping all
  //// the original types so it's more clear what is going on
  // pair<edge_iterator, edge_iterator> ei = edges(g);
  // for (edge_iterator edge_iter = ei.first; edge_iter != ei.second;
  //++edge_iter) {
  // cout << "(" << source(*edge_iter, g) << ", " << target(*edge_iter, g)
  //<< ")\n";
  //}

  // cout << "\n";

  //// Print out the edge list again to see that it has been added
  // for (edge_iterator edge_iter = ei.first; edge_iter != ei.second;
  //++edge_iter) {
  // cout << "(" << source(*edge_iter, g) << ", " << target(*edge_iter, g)
  //<< ")\n";
  //}

  ////...and print out our edge set once more to see that it was added
  // for (edge_iterator edge_iter = ei.first; edge_iter != ei.second;
  //++edge_iter) {
  // cout << "(" << source(*edge_iter, g) << ", " << target(*edge_iter, g)
  //<< ")\n";
  //}
  return 0;
}
