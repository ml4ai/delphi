#pragma once

#include <iostream>
#include <numeric>
#include <map>
#include <vector>
#include "DiGraph.hpp"

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
  typedef std::multimap<std::pair<int, int>, double*>::iterator MMAPIterator;

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
  std::multimap<std::pair<int, int>, double*> beta2product;

  public:
  Tran_Mat_Cell(int source, int target);

  // Add a path that starts with the start vertex and ends with the end vertex.
  bool add_path(std::vector<int>& path);

  // Allocates the prodcut std::vector with the same length as the paths
  // std::vector Populates the beat2prodcut multimap linking each β (edge - A
  // pair) to all the products that depend on it. This **MUST** be called after
  // adding all the paths usign add_path(). After populating the beta2product
  // multimap, the length of the products std::vector **MUST NOT** be changed.
  // If it is changes, we run into the danger of OS moving the products
  // std::vector into a different location in memory and pointers kept in
  // beta2product multimap becoming dangling pointer.
  void allocate_datastructures();

  // Computes the value of this cell from scratch.
  // Should be called after adding all the paths using add_path()
  // and calling allocate_datastructures()
  // TODO: This is just a dummy implementation. Update the logic to
  // calculate the value using βs assigned to each path.
  // To access βs, the graph needs to be passed in as an argument.
  // Logic is similar to Tran_Mat_Cell::sample_from_prior()
  // At the moment just compute the sum of lengths of all the paths
  double compute_cell(const DiGraph& CAG);

  double sample_from_prior(const DiGraph& CAG, int samp_num = 0);

  // Given a β and an update amount, update all the products where β is a
  // factor. compute_cell() must be called once at the beginning befor calling
  // this.
  double update_cell(std::pair<int, int> beta, double amount);

  void print_products();

  void print_beta2product();

  // Given an edge (source, target vertex ids - a β=∂target/∂source),
  // print all the products that are dependent on it.
  void print(int source, int target);

  void print_paths();

  void get_paths_shorter_than_or_equal_to(int length, bool from_beginning);

  std::unordered_set<int> get_vertices_within_hops(int hops,
                                                   bool from_beginning);

  std::unordered_set<int>
  get_vertices_on_paths_shorter_than_or_equal_to(int hops);

  bool has_multiple_paths_longer_than_or_equal_to(int length);
};
