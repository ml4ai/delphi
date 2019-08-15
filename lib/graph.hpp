#pragma once

#include <boost/graph/adjacency_list.hpp>

/** A collection of template functions to work with Boost Graph Library
 * graphs. */

namespace delphi::graph {
template <class G> auto vertices(G g) {
  return boost::make_iterator_range(boost::vertices(g));
}

template <class G, class V> auto successors(G g, V v) {
  return make_iterator_range(boost::adjacent_vertices(v, g));
}

template <class G> size_t size(G g) { return boost::num_vertices(g); }

template <class G> void add_edge(G g, int i, int j) {
  boost::add_edge(i, j, g);
}

template <class G> auto add_node(G g) { return boost::add_vertex(g); }
} // namespace delphi::graph


