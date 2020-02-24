#include "AnalysisGraph.hpp"
#include "Node.hpp"
#include "utils.hpp"
#include <boost/range/algorithm/for_each.hpp>
#include <range/v3/all.hpp>

using boost::for_each;
using delphi::utils::in;
using fmt::print;

using namespace std;

void AnalysisGraph::get_subgraph(int vert,
                                 unordered_set<int>& vertices_to_keep,
                                 int cutoff,
                                 bool inward) {

  // Mark the current vertex visited
  (*this)[vert].visited = true;
  vertices_to_keep.insert(vert);

  if (cutoff != 0) {
    cutoff--;

    // Recursively process all the vertices adjacent to the current vertex
    if (inward) {
      for_each(this->predecessors(vert), [&](int v) {
        if (!(*this)[v].visited) {
          this->get_subgraph(v, vertices_to_keep, cutoff, inward);
        }
      });
    }
    else {
      for_each(this->successors(vert), [&](int v) {
        if (!(*this)[v].visited) {
          this->get_subgraph(v, vertices_to_keep, cutoff, inward);
        }
      });
    }
  }

  // Mark the current vertex unvisited
  (*this)[vert].visited = false;
};

void AnalysisGraph::get_subgraph_between(int start,
                                         int end,
                                         vector<int>& path,
                                         unordered_set<int>& vertices_to_keep,
                                         int cutoff) {

  // Mark the current vertex visited
  (*this)[start].visited = true;

  // Add this vertex to the path
  path.push_back(start);

  // If current vertex is the destination vertex, then
  //   we have found one path.
  //   Add this cell to the Tran_Mat_Object that is tracking
  //   this transition matrix cell.
  if (start == end) {
    vertices_to_keep.insert(path.begin(), path.end());
  }
  else if (cutoff != 0) {
    cutoff--;

    // Recursively process all the vertices adjacent to the current vertex
    for_each(this->successors(start), [&](int v) {
      if (!(*this)[v].visited) {
        this->get_subgraph_between(v, end, path, vertices_to_keep, cutoff);
      }
    });
  }

  // Remove current vertex from the path and make it unvisited
  path.pop_back();
  (*this)[start].visited = false;
};

AnalysisGraph AnalysisGraph::get_subgraph_for_concept(string concept,
                                                      bool inward,
                                                      int depth) {

  using ranges::views::filter, ranges::views::transform, ranges::to;

  // Mark all the vertices as not visited
  for_each(this->nodes(), [](Node& node) { node.visited = false; });

  int num_verts = this->num_vertices();

  unordered_set<int> vertices_to_keep = unordered_set<int>();

  this->get_subgraph(
      this->get_vertex_id(concept), vertices_to_keep, depth, inward);

  unordered_set<string> nodes_to_remove =
      this->node_indices() |
      filter([&](int v) { return !in(vertices_to_keep, v); }) |
      transform([&](int v) { return (*this)[v].name; }) | to<unordered_set>();

  if (vertices_to_keep.size() == 0) {
    print("Subgraph has 0 nodes - returning an empty CAG!");
  }

  // Make a copy of current AnalysisGraph
  // TODO: We have to make sure that we are making a deep copy.
  //       Test so far does not show suspicious behavior
  AnalysisGraph G_sub = *this;
  for_each(nodes_to_remove, [&](string n) { G_sub.remove_node(n); });
  G_sub.clear_state();

  return G_sub;
}

AnalysisGraph AnalysisGraph::get_subgraph_for_concept_pair(
    string source_concept, string target_concept, int cutoff) {
  int src_id = this->get_vertex_id(source_concept);
  int tgt_id = this->get_vertex_id(target_concept);

  unordered_set<int> vertices_to_keep;
  unordered_set<string> vertices_to_remove;
  vector<int> path;

  // Mark all the vertices are not visited
  for_each(this->node_indices(), [&](int v) { (*this)[v].visited = false; });

  this->get_subgraph_between(src_id, tgt_id, path, vertices_to_keep, cutoff);

  if (vertices_to_keep.size() == 0) {
    print("AnalysisGraph::get_subgraph_for_concept_pair(): "
         "There are no paths of length <= {0} from "
         "source concept {1} --to-> target concept {2}. "
         "Returning an empty CAG!",
         cutoff,
         source_concept,
         target_concept);
  }

  // Determine the vertices to be removed
  for (int vert_id : this->node_indices()) {
    if (!in(vertices_to_keep, vert_id)) {
      vertices_to_remove.insert((*this)[vert_id].name);
    }
  }

  // Make a copy of current AnalysisGraph
  // TODO: We have to make sure that we are making a deep copy.
  //       Test so far does not show suspicious behavior
  AnalysisGraph G_sub = *this;
  G_sub.remove_nodes(vertices_to_remove);
  G_sub.clear_state();

  return G_sub;
}
