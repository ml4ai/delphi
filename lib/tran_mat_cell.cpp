#include <range/v3/all.hpp>
#include <tran_mat_cell.hpp>

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

  for (int p = 0; p < this->paths.size(); p++) {
    for (int v = 0; v < this->paths[p].size() - 1; v++) {
      // Each β along this path is a factor of the product of this path.
      this->beta2product.insert(
          make_pair(make_pair(paths[p][v], paths[p][v + 1]), &products[p]));
    }
  }
}

// Computes the value of this cell from scratch.
// Should be called after adding all the paths using add_path()
// and calling allocate_datastructures()
// TODO: This is just a dummy implementation. Update the logic to
// calculate the value using βs assigned to each path.
// To access βs, the graph needs to be passed in as an argument.
// Logic is similar to Tran_Mat_Cell::sample_from_prior()
// At the moment just compute the sum of lengths of all the paths
double Tran_Mat_Cell::compute_cell(const DiGraph& CAG) {
  for (int p = 0; p < this->paths.size(); p++) {
    this->products[p] = 1; // 0;

    for (int v = 0; v < this->paths[p].size() - 1; v++) {
      auto edg = edge(paths[p][v], paths[p][v + 1], CAG);
      const double& beta = CAG[edg.first].beta;

      this->products[p] *= beta; //+= 1;
    }
  }

  return rs::accumulate(products, 0.0);
}

double Tran_Mat_Cell::sample_from_prior(const DiGraph& CAG, int samp_num) {
  for (int p = 0; p < this->paths.size(); p++) {
    this->products[p] = 1;

    // Assume that none of the edges along this path has KDEs assigned.
    // At the end of traversing this path, if that is the case, leaving
    // the product for this path as 1 does not seem correct. In this case
    // I feel that the best option is to make the product for this path 0.
    bool hasKDE = false;

    for (int v = 0; v < this->paths[p].size() - 1; v++) {
      const vector<double>& samples =
          CAG[edge(v, v + 1, CAG).first].kde.dataset;

      // Check whether we have enough samples to fulfil this request
      if (samples.size() > samp_num) {
        this->products[p] *=
            CAG[edge(v, v + 1, CAG).first].kde.dataset[samp_num];
      }
    }

    // If none of the edges along this path had a KDE assigned,
    // make the contribution of this path to the value of the cell 0.
    if (!hasKDE) {
      this->products[p] = 0;
    }
  }

  return rs::accumulate(products, 0.0);
}

// Given a β and an update amount, update all the products where β is a
// factor. compute_cell() must be called once at the beginning befor calling
// this.
double Tran_Mat_Cell::update_cell(pair<int, int> beta, double amount) {
  pair<MMAPIterator, MMAPIterator> res = this->beta2product.equal_range(beta);

  for (MMAPIterator it = res.first; it != res.second; it++) {
    *(it->second) *= amount;
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
