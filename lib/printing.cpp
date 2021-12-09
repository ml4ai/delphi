#include "AnalysisGraph.hpp"
#include <boost/range/algorithm/for_each.hpp>

using namespace std;
using fmt::print;


/*
 ============================================================================
 Public: Printing
 ============================================================================
*/

void AnalysisGraph::print_nodes() {
  print("Vertex IDs and their names in the CAG\n");
  print("Vertex ID : Name\n");
  print("--------- : ----\n");
  for_each(this->node_indices(), [&](int v) {
    cout << v << "         " << this->graph[v].name << endl;
  });

  print("Independent nodes\n");
  for (int v : this->head_nodes) {
    cout << v << "         " << this->graph[v].name << endl;
  }

  print("Dependent nodes\n");
  for (int v : this->body_nodes) {
    cout << v << "         " << this->graph[v].name << endl;
  }
}

void AnalysisGraph::print_edges() {
  for_each(edges(), [&](auto e) {
    cout << "(" << (*this)[boost::source(e, this->graph)].name << ", "
         << (*this)[boost::target(e, this->graph)].name << ")" << " - "
         << (this->graph[e].is_frozen()
                        ? "Frozen at " + to_string(this->graph[e].get_theta())
                        : "Free") << endl;
  });
}

void AnalysisGraph::print_name_to_vertex() {
  for (auto [name, vert] : this->name_to_vertex) {
    cout << name << " -> " << vert << endl;
  }
  cout << endl;
}

void AnalysisGraph::print_indicators() {
  cout << "-----Indicators-----\n";
  cout << "Indicators attached to nodes\n";
  for (int v : this->node_indices()) {
    cout << v << ":" << (*this)[v].name << endl;
    for (auto [name, vert] : (*this)[v].nameToIndexMap) {
      cout << "\t"
           << "indicator " << vert << ": " << name << endl;
    }
  }

  cout << "\nIndicators in CAG \n";
  for (string ind : this->indicators_in_CAG) {
      cout << "indicator: " << ind << endl;
  }
}

void AnalysisGraph::print_A_beta_factors() {
  int num_verts = this->num_vertices();

  for (int row = 0; row < num_verts; ++row) {
    for (int col = 0; col < num_verts; ++col) {
      cout << endl << "Printing cell: (" << row << ", " << col << ") " << endl;
      if (this->A_beta_factors[row][col]) {
        this->A_beta_factors[row][col]->print_beta2product();
      }
    }
  }
}

void AnalysisGraph::print_latent_state(const Eigen::VectorXd& v) {
  for (int i=0; i < this->num_vertices(); i++){
    cout << (*this)[i].name << " " << v[2*i] << endl;
  }
}

void AnalysisGraph::print_all_paths() {
  int num_verts = this->num_vertices();

//  if (this->A_beta_factors.size() != num_verts ||
//      this->A_beta_factors[0].size() != num_verts) {
//    this->find_all_paths();
//  }
  this->find_all_paths();

  cout << "All the simple paths of:" << endl;

  for (int row = 0; row < num_verts; ++row) {
    for (int col = 0; col < num_verts; ++col) {
      if (this->A_beta_factors[row][col]) {
        this->A_beta_factors[row][col]->print_paths();
      }
    }
  }
}

// Given an edge (source, target vertex ids - i.e. a β ≡ ∂target/∂source),
// print all the transition matrix cells that are dependent on it.
void AnalysisGraph::print_cells_affected_by_beta(int source, int target) {
  typedef multimap<pair<int, int>, pair<int, int>>::iterator MMapIterator;

  pair<int, int> beta = make_pair(source, target);

  pair<MMapIterator, MMapIterator> beta_dept_cells =
      this->beta2cell.equal_range(beta);

  cout << endl
       << "Cells of A affected by beta_(" << source << ", " << target << ")"
       << endl;

  for (MMapIterator it = beta_dept_cells.first; it != beta_dept_cells.second;
       it++) {
    cout << "(" << it->second.first * 2 << ", " << it->second.second * 2 + 1
         << ") ";
  }
  cout << endl;
}

void AnalysisGraph::print_training_range() {
    std::cout << "ID         : " << this->id << std::endl;
    std::cout << "Start year : " << this->training_range.first.first << std::endl;
    std::cout << "Start month: " << this->training_range.first.second << std::endl;
    std::cout << "End year   : " << this->training_range.second.first << std::endl;
    std::cout << "End month  : " << this->training_range.second.second << std::endl;
}
