#pragma once

#include <kde.hpp>
#include <optional>

template <class T> void print(T x) { std::cout << x << std::endl; }

template <class T> void printVec(std::vector<T> xs) {
  for (auto x : xs) {
    print(x);
  }
}

struct CausalFragment {
  std::string subj_adjective;
  std::string obj_adjective;
  // Here we assume that unknown polarities are set to 1.
  int subj_polarity{1};
  int obj_polarity{1};
};

struct Edge {
  std::string name;
  std::optional<KDE> kde;
  std::vector<CausalFragment> causalFragments = {};
};

struct Node {
  std::string name;
  bool visited;

  // Stores all the simple directed paths ending at this node
  // according to the starting vertex of each path
  // used as the key of the map.
  // start --> [ path1, path2, path3 ]
  // path = [ (start, v2), (v2, v3), (v3, this_node) ]
  std::unordered_map<int, std::vector<vector<std::pair<int, int>>>>
      influenced_by;
};

struct GraphData {
  std::string name;
};

typedef boost::adjacency_list<boost::setS,
                              boost::vecS,
                              boost::bidirectionalS,
                              Node,
                              Edge,
                              GraphData>
    DiGraph;
