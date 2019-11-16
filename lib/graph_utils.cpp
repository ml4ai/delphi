#include "AnalysisGraph.hpp"
#include <boost/range/algorithm/for_each.hpp>

using namespace std;


/*
 ============================================================================
 Private: Utilities
 ============================================================================
*/

void AnalysisGraph::clear_state() {
  this->A_beta_factors.clear();

  // Clear the multimap that keeps track of cells in the transition
  // matrix that are dependent on each β.
  this->beta2cell.clear();

  // Clear the set of all the β dependent cells
  this->beta_dependent_cells.clear();
}

void AnalysisGraph::initialize_random_number_generator() {
  // Define the random number generator
  // All the places we need random numbers share this generator

  this->rand_num_generator = RNG::rng()->get_RNG();

  // Uniform distribution used by the MCMC sampler
  this->uni_dist = uniform_real_distribution<double>(0.0, 1.0);

  // Normal distrubution used to perturb β
  this->norm_dist = normal_distribution<double>(0.0, 1.0);
}

void AnalysisGraph::remove_node(int node_id) {
  // Delete all the edges incident to this node
  clear_vertex(node_id, this->graph);

  // Remove the vertex
  remove_vertex(node_id, this->graph);

  // Update the internal meta-data
  for (int vert_id : this->node_indices()) {
    this->name_to_vertex.at((*this)[vert_id].name) = vert_id;
  }
}

void AnalysisGraph::allocate_A_beta_factors() {
  this->A_beta_factors.clear();

  int num_verts = this->num_vertices();

  for (int vert = 0; vert < num_verts; ++vert) {
    this->A_beta_factors.push_back(
        vector<shared_ptr<Tran_Mat_Cell>>(num_verts));
  }
}

void AnalysisGraph::find_all_paths_between(int start,
                                           int end,
                                           int cutoff = -1) {
  // Mark all the vertices are not visited
  for_each(this->node_indices(), [&](int v) { (*this)[v].visited = false; });

  // Create a vector of ints to store paths.
  vector<int> path;

  this->find_all_paths_between_util(start, end, path, cutoff);
}

void AnalysisGraph::find_all_paths_between_util(int start,
                                                int end,
                                                vector<int>& path,
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
    // Add this path to the relevant transition matrix cell
    if (!A_beta_factors[path.back()][path[0]]) {
      this->A_beta_factors[path.back()][path[0]].reset(
          new Tran_Mat_Cell(path[0], path.back()));
    }

    this->A_beta_factors[path.back()][path[0]]->add_path(path);

    // This transition matrix cell is dependent upon Each β along this path.
    pair<int, int> this_cell = make_pair(path.back(), path[0]);

    beta_dependent_cells.insert(this_cell);

    for (int v = 0; v < path.size() - 1; v++) {
      this->beta2cell.insert(
          make_pair(make_pair(path[v], path[v + 1]), this_cell));
    }
  }
  else if (cutoff != 0) {
    cutoff--;
    // Current vertex is not the destination
    // Recursively process all the vertices adjacent to the current vertex
    for_each(this->successors(start), [&](int v) {
      if (!(*this)[v].visited) {
        this->find_all_paths_between_util(v, end, path, cutoff);
      }
    });
  }

  // Remove current vertex from the path and make it unvisited
  path.pop_back();
  (*this)[start].visited = false;
}

int AnalysisGraph::calculate_num_timesteps(int start_year,
                                           int start_month,
                                           int end_year,
                                           int end_month) {
  assert(start_year <= end_year);

  if (start_year == end_year) {
    assert(start_month <= end_month);
  }

  int diff_year = end_year - start_year;
  int year_to_month = diff_year * 12;

  // NOTE: I am adding 1 here itself so that I do not have to add it outside
  return year_to_month - (start_month - 1) + (end_month - 1) + 1;
}


/*
 ============================================================================
 Public: Utilities
 ============================================================================
*/

void AnalysisGraph::find_all_paths() {
  auto verts = this->node_indices();

  // Allocate the 2D array that keeps track of the cells of the transition
  // matrix (A_original) that are dependent on βs.
  // This function can be called anytime after creating the CAG.
  this->allocate_A_beta_factors();

  // Clear the multimap that keeps track of cells in the transition
  // matrix that are dependent on each β.
  this->beta2cell.clear();

  // Clear the set of all the β dependent cells
  this->beta_dependent_cells.clear();

  for_each(verts, [&](int start) {
    for_each(verts, [&](int end) {
      if (start != end) {
        this->find_all_paths_between(start, end);
      }
    });
  });

  // Allocate the cell value calculation data structures
  int num_verts = this->num_vertices();

  for (int row = 0; row < num_verts; ++row) {
    for (int col = 0; col < num_verts; ++col) {
      if (this->A_beta_factors[row][col]) {
        this->A_beta_factors[row][col]->allocate_datastructures();
      }
    }
  }
}

void AnalysisGraph::set_random_seed(int seed) {
  this->rng_instance = RNG::rng();
  this->rng_instance->set_seed(seed);
}

void AnalysisGraph::set_derivative(string concept, double derivative) {
  int v = this->get_vertex_id(concept);
  this->s0[2 * v + 1] = derivative;
}
