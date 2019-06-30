#pragma once

#include <nlohmann/json.hpp>
#include <kde.hpp>
#include <optional>

using std::pair 
    , std::string
    , std::optional
    , std::vector
    , std::unordered_map
    , boost::adjacency_list
    , boost::vecS
    , boost::setS
    , boost::listS
    , boost::bidirectionalS
    , boost::range::for_each
;


template <class T>
void print(T x) {
  std::cout << x << std::endl;
}

template <class T>
void printVec(vector<T> xs) {
  for (auto x : xs) {print(x);}
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
  vector<CausalFragment> causalFragments= {};
};

struct Node {
  string name;
  bool visited;
  vector< vector< pair< int, int >>> influences;
};

struct GraphData {
  string name;
};


typedef adjacency_list<setS, vecS, bidirectionalS, Node, Edge, GraphData> DiGraph;
