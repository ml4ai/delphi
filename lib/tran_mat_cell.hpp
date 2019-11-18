#pragma once

#include <iostream>
#include <map>
#include <numeric>
#include <vector>

/**
 * This class represents a single cell of the transition matrix which is
 * computed by a sum of products of βs. Accordign to our current model, which
 * uses variables and their partial derivatives with respect to each other ( x
 * --> y, βxy = ∂y/∂x ), at most half of the transition matrix cells are
 * affected by βs. According to the way we organize the transition matrix, the
 * cells A[row][col] where row is an even index and col is an odd index are such
 * cells.
 */
class Tran_Mat_Cell {
private:
  typedef std::multimap<std::pair<int, int>, double *>::iterator MMAPIterator;

  // All the directed paths in the CAG that starts at source vertex and ends
  // at target vertex decides the value of the transition matrix cell
  // A[ 2 * source ][ 2 * target + 1 ]
  int source;
  int target;

  // Records all the directed paths in the CAG that
  // starts at source vertex and ends at target vertex.
  // Each path informs how to calculate one product in the calculation of the
  // value of this transition matrix cell, which is a sum of products.
  // We multiply all the βs along a path to compute a product. Then we add all
  // the products to compute the value of the cell.
  // TODO: βs can be very small numbers. Multiplying a bunch of them could
  // run into floating point underflow. Can we store log( β )s instead of
  // βs and add them and then take the exp of the addition?
  std::vector<std::vector<int>> paths;

  // Keeps the value of each product. There is an entry for each path here.
  // So, there is a 1-1 mapping between the two std::vectors paths and products.
  std::vector<double> products;

  // Maps each β to all the products where that β is a factor. This mapping
  // is needed to quickly update the products and the cell value upon
  // purturbing one β.
  std::multimap<std::pair<int, int>, double *> beta2product;

public:
  Tran_Mat_Cell(int source, int target) {
    this->source = source;
    this->target = target;
  }

  // Add a path that starts with the start vertex and ends with the end vertex.
  bool add_path(std::vector<int> &path) {
    if (path[0] == this->source && path.back() == this->target) {
      this->paths.push_back(path);
      return true;
    }

    return false;
  }

  // Allocates the prodcut std::vector with the same length as the paths
  // std::vector Populates the beat2prodcut multimap linking each β (edge - A
  // pair) to all the products that depend on it. This **MUST** be called after
  // adding all the paths usign add_path(). After populating the beta2product
  // multimap, the length of the products std::vector **MUST NOT** be changed.
  // If it is changes, we run into the danger of OS moving the products
  // std::vector into a different location in memory and pointers kept in
  // beta2product multimap becoming dangling pointer.
  void allocate_datastructures() {
    // TODO: Decide the correct initial value
    this->products = std::vector<double>(paths.size(), 0);
    this->beta2product.clear();

    for (int p = 0; p < this->paths.size(); p++) {
      for (int v = 0; v < this->paths[p].size() - 1; v++) {
        // Each β along this path is a factor of the product of this path.
        this->beta2product.insert(std::make_pair(
            std::make_pair(paths[p][v], paths[p][v + 1]), &products[p]));
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
  double compute_cell(const DiGraph &CAG) {
    for (int p = 0; p < this->paths.size(); p++) {
      this->products[p] = 1; // 0;

      for (int v = 0; v < this->paths[p].size() - 1; v++) {
        auto edg = boost::edge(paths[p][v], paths[p][v + 1], CAG);
        const double &beta = CAG[edg.first].beta;

        this->products[p] *= beta; //+= 1;
      }
    }

    return accumulate(products.begin(), products.end(), 0.0);
  }

  double sample_from_prior(const DiGraph &CAG, int samp_num = 0) {
    for (int p = 0; p < this->paths.size(); p++) {
      this->products[p] = 1;

      // Assume that none of the edges along this path has KDEs assigned.
      // At the end of traversing this path, if that is the case, leaving
      // the product for this path as 1 does not seem correct. In this case
      // I feel that the best option is to make the product for this path 0.
      bool hasKDE = false;

      for (int v = 0; v < this->paths[p].size() - 1; v++) {
        // Check wheather this edge has a KDE assigned to it
        // TODO: Why does KDE of an edge is optional?
        // I guess, there could be edges that do not have a KDE assigned.
        // What should we do when one of more edges along a path does not have
        // a KDE assigned?
        // At the moment, the code silently skips that edge as if that edge
        // does not exist in the path. Is this the correct thing to do?
        // What happens when all the edges of a path do not have KDEs assigned?
          // Vector of all the samples for this edge
          const std::vector<double> &samples =
              CAG[boost::edge(v, v + 1, CAG).first].kde.dataset;

          // Check whether we have enough samples to fulfil this request
          if (samples.size() > samp_num) {
            this->products[p] *= CAG[boost::edge(v, v + 1, CAG).first]
                                     .kde.dataset[samp_num];
        }
      }

      // If none of the edges along this path had a KDE assigned,
      // make the contribution of this path to the value of the cell 0.
      if (!hasKDE) {
        this->products[p] = 0;
      }
    }

    return accumulate(products.begin(), products.end(), 0.0);
  }

  // Given a β and an update amount, update all the products where β is a
  // factor. compute_cell() must be called once at the beginning befor calling
  // this.
  double update_cell(std::pair<int, int> beta, double amount) {
    std::pair<MMAPIterator, MMAPIterator> res =
        this->beta2product.equal_range(beta);

    for (MMAPIterator it = res.first; it != res.second; it++) {
      *(it->second) *= amount;
    }

    return accumulate(products.begin(), products.end(), 0.0);
  }

  void print_products() {
    for (double f : this->products) {
      std::cout << f << " ";
    }
    std::cout << std::endl;
  }

  void print_beta2product() {
    for (auto it = this->beta2product.begin(); it != this->beta2product.end();
         it++) {
      fmt::print("({}, {} -> {})", it->first.first, it->first.second,
                 *(it->second));
    }
  }

  // Given an edge (source, target vertex ids - a β=∂target/∂source),
  // print all the products that are dependent on it.
  void print(int source, int target) {
    std::pair<int, int> beta = std::make_pair(source, target);

    std::pair<MMAPIterator, MMAPIterator> res =
        this->beta2product.equal_range(beta);

    std::cout << "(" << beta.first << ", " << beta.second << ") -> ";
    for (MMAPIterator it = res.first; it != res.second; it++) {
      std::cout << *(it->second) << " ";
    }
    std::cout << std::endl;
  }

  void print_paths() {
    std::cout << std::endl
              << "Paths between vertices: " << this->source << " and "
              << this->target << std::endl;
    for (std::vector<int> path : this->paths) {
      for (int vert : path) {
        std::cout << vert << " -> ";
      }
      std::cout << std::endl;
    }
  }

  void get_paths_shorter_than_or_equal_to(int length, bool from_beginning) {
    std::cout << std::endl
              << "Paths between vertices: " << this->source << " and "
              << this->target << std::endl;

    if (from_beginning) {
      for (std::vector<int> path : this->paths) {
        for (std::vector<int>::iterator vert = path.begin();
             vert < path.end() && vert <= path.begin() + length; vert++) {
          std::cout << *vert << " -> ";
        }
        std::cout << std::endl;
      }
    } else {
      for (std::vector<int> path : this->paths) {
        std::vector<int>::iterator vert =
            path.size() <= length ? path.begin() : path.end() - length - 1;
        for (; vert < path.end(); vert++) {
          std::cout << *vert << " -> ";
        }
        std::cout << std::endl;
      }
    }
  }

  std::unordered_set<int> get_vertices_within_hops(int hops,
                                                   bool from_beginning) {

    std::unordered_set<int> vertices_within_hops;

    if (from_beginning) {
      for (std::vector<int> path : this->paths) {
        for (std::vector<int>::iterator vert = path.begin();
             vert < path.end() && vert <= path.begin() + hops; vert++) {
          // std::cout << *vert << " -> ";
          vertices_within_hops.insert(*vert);
        }
        // std::cout << std::endl;
      }
    } else {
      for (std::vector<int> path : this->paths) {
        std::vector<int>::iterator vert =
            path.size() <= hops ? path.begin() : path.end() - hops - 1;
        for (; vert < path.end(); vert++) {
          // std::cout << *vert << " -> ";
          vertices_within_hops.insert(*vert);
        }
        // std::cout << std::endl;
      }
    }

    return vertices_within_hops;
  }

  std::unordered_set<int>
  get_vertices_on_paths_shorter_than_or_equal_to(int hops) {

    std::unordered_set<int> vertices_on_shorter_paths;

    for (std::vector<int> path : this->paths) {
      if (path.size() <= hops + 1) {
        for (std::vector<int>::iterator vert = path.begin(); vert < path.end();
             vert++) {
          // std::cout << *vert << " -> ";
          vertices_on_shorter_paths.insert(*vert);
        }
        // std::cout << std::endl;
      }
    }

    return vertices_on_shorter_paths;
  }

  bool has_multiple_paths_longer_than_or_equal_to(int length) {
    int longer_path_count = 0;

    for (std::vector<int> path : this->paths) {
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
};
