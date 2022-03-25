#include <range/v3/all.hpp>
#include <Tran_Mat_Cell.hpp>

using namespace std;
namespace rs = ranges;
using boost::edge;

Tran_Mat_Cell::Tran_Mat_Cell(int source, int target) {
  this->source = source;
  this->target = target;
}

// Add a path that starts with the start vertex and ends with the end vertex.
bool Tran_Mat_Cell::add_path(vector<int>& path) {
  if (path[0] == this->source && path.back() == this->target) {
    this->paths.push_back(path);
    return true;
  }

  return false;
}

void Tran_Mat_Cell::allocate_datastructures() {
  // TODO: Decide the correct initial value
  this->products = vector<double>(paths.size(), 0);
  this->beta2product.clear();

  for (unsigned long int p = 0; p < this->paths.size(); p++) {
    for (unsigned long int v = 0; v < this->paths[p].size() - 1; v++) {
      // Each β along this path is a factor of the product of this path.
      this->beta2product.insert(
          make_pair(make_pair(paths[p][v], paths[p][v + 1]), &products[p]));
    }
  }
}

// Computes the value of this cell from scratch.
// Should be called after adding all the paths using add_path()
// and calling allocate_datastructures()
//         θst
// ┏━━━━━━━━━━━━━━━━━┓
// ┃ θsx   θxy   θyt ↓
// 〇━━━>〇━━━>〇━━━>〇
// ┃  θsw      θwt   ↑
// ┗━━━━━━━>〇━━━━━━━┛
//
// We are computing
//   [tan(θst)] + [(tan(θsx) × tan(θxy) × tan(θyt)] + [tan(θsw) × tan(θwt)]
// In the transition matrix A, this cell is for all the paths starting at
// vertex s and ending at vertex t. If s and t are the indices of the
// respective vertexes, this cell is A[2 × t][2 × s + 1]
double Tran_Mat_Cell::compute_cell(const DiGraph& CAG) {
  for (unsigned long int p = 0; p < this->paths.size(); p++) {
    this->products[p] = 1; // 0;

    for (unsigned long int v = 0; v < this->paths[p].size() - 1; v++) {
      auto edg = edge(paths[p][v], paths[p][v + 1], CAG);
      // β = tan(θ)
      double beta = tan(CAG[edg.first].get_theta());

      this->products[p] *= beta; //+= 1;
    }
  }

  return rs::accumulate(products, 0.0);
}

void Tran_Mat_Cell::print_products() {
  for (double f : this->products) {
    cout << f << " ";
  }
  cout << endl;
}

void Tran_Mat_Cell::print_beta2product() {
  for (auto it = this->beta2product.begin(); it != this->beta2product.end();
       it++) {
    fmt::print(
        "({}, {} -> {})", it->first.first, it->first.second, *(it->second));
  }
}

// Given an edge (source, target vertex ids - a β=∂target/∂source),
// print all the products that are dependent on it.
void Tran_Mat_Cell::print(int source, int target) {
  pair<int, int> beta = make_pair(source, target);

  pair<MMAPIterator, MMAPIterator> res = this->beta2product.equal_range(beta);

  cout << "(" << beta.first << ", " << beta.second << ") -> ";
  for (MMAPIterator it = res.first; it != res.second; it++) {
    cout << *(it->second) << " ";
  }
  cout << endl;
}

void Tran_Mat_Cell::print_paths() {
  cout << endl
       << "Paths between vertices: " << this->source << " and " << this->target
       << endl;
  for (vector<int> path : this->paths) {
    for (int vert : path) {
      cout << vert << " -> ";
    }
    cout << endl;
  }
}

void Tran_Mat_Cell::get_paths_shorter_than_or_equal_to(int length,
                                                       bool from_beginning) {
  cout << endl
       << "Paths between vertices: " << this->source << " and " << this->target
       << endl;

  if (from_beginning) {
    for (vector<int> path : this->paths) {
      for (vector<int>::iterator vert = path.begin();
           vert < path.end() && vert <= path.begin() + length;
           vert++) {
        cout << *vert << " -> ";
      }
      cout << endl;
    }
  }
  else {
    for (vector<int> path : this->paths) {
      vector<int>::iterator vert =
          path.size() <= length ? path.begin() : path.end() - length - 1;
      for (; vert < path.end(); vert++) {
        cout << *vert << " -> ";
      }
      cout << endl;
    }
  }
}

unordered_set<int>
Tran_Mat_Cell::get_vertices_within_hops(int hops, bool from_beginning) {

  unordered_set<int> vertices_within_hops;

  if (from_beginning) {
    for (vector<int> path : this->paths) {
      for (vector<int>::iterator vert = path.begin();
           vert < path.end() && vert <= path.begin() + hops;
           vert++) {
        vertices_within_hops.insert(*vert);
      }
    }
  }
  else {
    for (vector<int> path : this->paths) {
      vector<int>::iterator vert =
          path.size() <= hops ? path.begin() : path.end() - hops - 1;
      for (; vert < path.end(); vert++) {
        vertices_within_hops.insert(*vert);
      }
    }
  }

  return vertices_within_hops;
}

unordered_set<int>
Tran_Mat_Cell::get_vertices_on_paths_shorter_than_or_equal_to(int hops) {

  unordered_set<int> vertices_on_shorter_paths;

  for (vector<int> path : this->paths) {
    if (path.size() <= hops + 1) {
      for (vector<int>::iterator vert = path.begin(); vert < path.end();
           vert++) {
        // cout << *vert << " -> ";
        vertices_on_shorter_paths.insert(*vert);
      }
      // cout << endl;
    }
  }

  return vertices_on_shorter_paths;
}

bool Tran_Mat_Cell::has_multiple_paths_longer_than_or_equal_to(int length) {
  int longer_path_count = 0;

  for (vector<int> path : this->paths) {
    // Note: A path records the sequence of nodes in the path
    //       e.g.: a -> b -> c-> d
    //       Therefore, the length of this path is
    //       path.size() - 1
    //       So, path.size() > length =>
    //           length of path >= length
    if (path.size() > length) {
      longer_path_count++;
    }
  }

  return longer_path_count > 1;
}
