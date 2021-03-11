#pragma once

#include <string>
#include <boost/graph/adjacency_list.hpp>
#include "Node.hpp"
#include "Edge.hpp"

class GraphData {
  public:
  std::string name;
};

//adjacency_list<OutEdgeList, VertexList, Directed,
//                VertexProperties, EdgeProperties,
//                GraphProperties, EdgeList>
// Note: When setS or hash_setS is used as OutEdgeList, Boost Graph Library
//       gives memory problems on remove_vertex cal sometimes leading to
// segmentation faults. So changed it to vecS
typedef boost::adjacency_list<boost::vecS,
                              boost::vecS,
                              boost::bidirectionalS,
                              Node,
                              Edge,
                              GraphData>
    DiGraph;
