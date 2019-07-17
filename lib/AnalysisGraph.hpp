#pragma once

#include <optional>
#include "kde.hpp"
#include "random_variables.hpp"

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

  // The current β for this edge
  // TODO: Need to decide how to initialize this or
  // decide whethr this is the correct way to do this.
  double beta = 1.0;
};

struct Node {
  std::string name = "";
  bool visited;
  LatentVar rv;

  std::vector< Indicator > indicators;
  // Maps each indicator name to its index in the indicators vector
  std::map< std::string, int > indicator_names;
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
