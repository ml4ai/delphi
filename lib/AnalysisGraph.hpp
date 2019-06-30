#pragma once

#include <kde.hpp>
#include <nlohmann/json.hpp>
#include <optional>

using std::pair, std::string, std::optional, std::vector, std::unordered_map;

template <class T> void print(T x) { std::cout << x << std::endl; }

template <class T> void printVec(vector<T> xs) {
  for (auto x : xs) {
    print(x);
  }
}

struct CausalFragment {
  string subj_adjective;
  string obj_adjective;
  // Here we assume that unknown polarities are set to 1.
  int subj_polarity = 1;
  int obj_polarity = 1;
};

struct Edge {
  string name;
  optional<KDE> kde;
  vector<CausalFragment> causalFragments = {};
};

struct Node {
  string name;
  bool visited;

  // Stores all the simple directed paths ending at this node
  // according to the starting vertex of each path
  // used as the key of the map.
  // start --> [ path1, path2, path3 ]
  // path = [ (start, v2), (v2, v3), (v3, this_node) ]
  unordered_map<int, vector<vector<pair<int, int>>>> influenced_by;
};

struct GraphData {
  string name;
};

typedef boost::adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS,
                              Node, Edge, GraphData>
    DiGraph;
