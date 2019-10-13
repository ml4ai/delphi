#pragma once

#include <string>
#include <boost/graph/adjacency_list.hpp>
#include "Node.hpp"
#include "Edge.hpp"

class GraphData {
  public:
  std::string name;
};

typedef boost::adjacency_list<boost::setS,
                              boost::vecS,
                              boost::bidirectionalS,
                              Node,
                              Edge,
                              GraphData>
    DiGraph;
