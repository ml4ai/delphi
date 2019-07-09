#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sqlite3.h>
#include <utility>
#include <Eigen/Dense>
#include <pybind11/eigen.h>

#include "cppitertools/itertools.hpp"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>

#include "AnalysisGraph.hpp"
#include "utils.hpp"

using std::cout, std::endl, std::unordered_map, std::pair, std::string,
    std::ifstream, std::stringstream, std::map, std::multimap, std::make_pair, std::set,
    boost::inner_product, boost::edge, boost::source, boost::target, boost::graph_bundle,
    boost::make_label_writer, boost::write_graphviz, boost::lambda::make_const,
    utils::load_json, utils::hasKey, utils::get, utils::lmap;


const size_t default_n_samples = 100;


/**
 * This class represents a single cell of the transition matrix which is computed
 * by a sum of products of βs.
 * Accordign to our current model, which uses variables and their partial
 * derivatives with respect to each other ( x --> y, βxy = ∂y/∂x ), only half of the
 * transition matrix cells are affected by βs. 
 * According to the way we organize the transition matrix, the cells A[row][col]
 * where row is an odd index and col is an even index are such cells.
 */
class Tran_Mat_Cell {
  private:
    typedef multimap< pair< int, int >, double * >::iterator MMAPIterator;

    // All the directed paths in the CAG that starts at source vertex and ends
    // at target vertex decides the value of the transition matrix cell
    // A[ 2 * source ][ 2 * target + 1 ]
    int source;
    int target;

    // Records all the directed paths in the CAG that starts at source vertex
    // and ends at target vertex.
    // Each path informs how to calculate one product in the calculation of the
    // value of this transition matrix cell, which is a sum of products.
    // We multiply all the βs along a path to compute a product. Then we add all
    // the products to compute the value of the cell.
    vector< vector< int >> paths;

    // Keeps the value of each product. There is an entry for each path here.
    // So, there is a 1-1 mapping between the two vectors paths and products.
    vector< double > products;

    // Maps each β to all the products where that β is a factor. This mapping
    // is needed to quickly update the products and the cell value upon
    // purturbing one β.
    multimap< pair< int, int >, double * > beta2product;

    
  public:
    Tran_Mat_Cell( int source, int target )
    {
      this->source = source;
      this->target = target;
    }


    // Add a path that starts with the start vertex and ends with the end vertex.
    bool add_path( vector< int > & path )
    {
      if( path[0] == this->source && path.back() == this->target )
      {
        this->paths.push_back( path );
        return true;
      }

      return false;
    }


    // Allocates the prodcut vector with the same length as the paths vector
    // Populates the beat2prodcut multimap linking each β (edge - A pair) to
    // all the products that depend on it.
    // This **MUST** be called after adding all the paths usign add_path.
    // After populating the beta2product multimap, the length of the products
    // vector **MUST NOT** be changed.
    // If it is changes, we run into the dange or OS moving the products vector
    // into a different location in memory and pointers kept in beta2product
    // multimap becoming dangling pointer.
    void allocate_datastructures()
    {
      // TODO: Decide the correct initial value
      this->products = vector< double >( paths.size(), 0 );
      this->beta2product.clear();

      for( int p = 0; p < this->paths.size(); p++ )
      {
        for( int v = 0; v < this->paths[p].size() - 1; v++ )
        {
          // Each β along this path is a factor of the product of this path.
          this->beta2product.insert( make_pair( make_pair( paths[p][v], paths[p][v+1] ), &products[p]));
        }
      }
    }
    

    // Computes the value of this cell from scratch.
    // Should be called after adding all the paths using add_path()
    // and calling allocate_datastructures()
    double compute_cell()
    {
      for( int p = 0; p < this->paths.size(); p++ )
      {
        for( int v = 0; v < this->paths[p].size() - 1; v++ )
        {
          this->products[p] += 1;
        }
      }

      return accumulate( products.begin(), products.end(), 0 );
    }


   double sample_from_prior( const DiGraph &  CAG, int samp_num )
    {
      for( int p = 0; p < this->paths.size(); p++ )
      {
        this->products[p] = 1;

        // Assume that none of the edges along this path has KDEs assigned.
        // At the end of traversing this path, if that is the case, leaving
        // the product for this path as 1 does not seem correct. In this case
        // I feel that that best option is to make the product for this path 0.
        bool hasKDE = false;

        for( int v = 0; v < this->paths[p].size() - 1; v++ )
        {
          // Check wheather this edge has a KDE assigned to it
          // TODO: Why does KDE of an edge is optional?
          // I guess, there could be edges that do not have a KDE assigned.
          // What should we do when one of more edges along a path does not have
          // a KDE assigned?
          // At the moment, the code silently skips that edge as if that edge
          // does not exist in the path. Is this the correct thing to do?
          // What happens when all the edges of a path do not have KDEs assigned?
          if( CAG[boost::edge( v, v+1, CAG ).first].kde.has_value()) 
          {
            // Vector of all the samples for this edge
            const vector<double> & samples = CAG[boost::edge( v, v+1, CAG ).first].kde.value().dataset;
            
            // Check whether we have enough samples to fulfil this request
            if( samples.size() > samp_num )
            {
              this->products[p] *= CAG[boost::edge( v, v+1, CAG ).first].kde.value().dataset[ samp_num ];
              hasKDE = true;
            }
            else
            {
              // What should we do if there is not enough samples generated for this path?
            }
          }
        }

        // If none of the edges along this path had a KDE assigned,
        // make the contribution of this path to the value of the cell 0.
        if( ! hasKDE )
        {
          this->products[p] = 0;
        }
      }

      return accumulate( products.begin(), products.end(), 0 );
    }


    // Given a β and an update amount, update all the products where β is a factor.
    // compute_cell() must be called once at the beginning befor calling this.
    double update_cell( pair< int, int > beta, double amount )
    {
      pair<MMAPIterator, MMAPIterator> res = this->beta2product.equal_range( beta );

      for( MMAPIterator it = res.first; it != res.second; it++ )
      {
        *(it->second) *= amount;
      }

      return accumulate( products.begin(), products.end(), 0 );
    }


    void print_products()
    {
      for( double f : this->products )
      {
        cout << f << " ";
      }
      cout << endl;
    }


    void print_beta2product()
    {
      for(auto it = this->beta2product.begin(); it != this->beta2product.end(); it++ )
      {
        cout << "(" << it->first.first << ", " << it->first.second << ") -> " << *(it->second) << endl;
      }

    }


    // Given an edge (source, target vertex ids - a β=∂target/∂source), 
    // print all the products that are dependent on it.
    void print( int source, int target )
    {
      pair< int, int > beta = make_pair( source, target );

      pair<MMAPIterator, MMAPIterator> res = this->beta2product.equal_range( beta );

      cout << "(" << beta.first << ", " << beta.second << ") -> ";
      for( MMAPIterator it = res.first; it != res.second; it++ )
      {
        cout << *(it->second) << " ";
      }
      cout << endl;
    }


    void print_paths()
    {
      cout << endl << "Paths between vectices: " << this->source << " and " << this->target << endl;
      for( vector< int > path : this->paths )
      {
        for( int vert : path )
        {
          cout << vert << " -> ";
        }
        cout << endl;
      }
    }
};



unordered_map<string, vector<double>>
construct_adjective_response_map(size_t n_kernels = default_n_samples) {
  sqlite3 *db;
  int rc = sqlite3_open(std::getenv("DELPHI_DB"), &db);
  if (!rc)
    print("Opened db successfully");
  else
    print("Could not open db");

  sqlite3_stmt *stmt;
  const char *query = "select * from gradableAdjectiveData";
  rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
  unordered_map<string, vector<double>> adjective_response_map;

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    string adjective = std::string(
        reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2)));
    double response = sqlite3_column_double(stmt, 6);
    if (hasKey(adjective_response_map, adjective)) {
      adjective_response_map[adjective] = {response};
    } else {
      adjective_response_map[adjective].push_back(response);
    }
  }

  for (auto &[k, v] : adjective_response_map) {
    v = KDE(v).resample(n_kernels);
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return adjective_response_map;
}



/**
 * The AnalysisGraph class is the main model/interface for Delphi.
 */
class AnalysisGraph {
  DiGraph graph;

public:
  // Manujinda: I had to move this up since I am usign this within the private:
  // block This is ugly. We need to re-factor the code to make it pretty again
  auto vertices() { return boost::make_iterator_range(boost::vertices(graph)); }


  auto successors(int i) {
    return boost::make_iterator_range(boost::adjacent_vertices(i, graph));
  }


  // Allocate a num_verts x num_verts 2D array (vector of vectors)
  void allocate_A_beta_factors()
  {
    int num_verts = boost::num_vertices( graph );

    for( int vert = 0; vert < num_verts; ++vert )
    {
      this->A_beta_factors.push_back( vector< Tran_Mat_Cell * >( num_verts ));
    }

    cout << A_beta_factors.size() << endl;
    cout << A_beta_factors[0].size() << endl;
  }


  void print_A_beta_factors()
  {
    int num_verts = boost::num_vertices( graph );

    for( int row = 0; row < num_verts; ++row )
    {
      for( int col = 0; col < num_verts; ++col )
      {
        cout << endl << "Printing cell: (" << row << ", " << col << ") " << endl;
        if( this->A_beta_factors[ row ][ col ] != nullptr )
        {
          this->A_beta_factors[ row ][ col ]->print_beta2product();
        }
      }
    }
  }


private:

  // A_beta_factors is a 2D array (vector of vectors)
  // that keeps track of the beta factors involved with
  // each cell of the transition matrix A.
  // 
  // Accordign to our current model, which uses variables and their partial
  // derivatives with respect to each other ( x --> y, βxy = ∂y/∂x ), only half of the
  // transition matrix cells are affected by βs. 
  // According to the way we organize the transition matrix, the cells A[row][col]
  // where row is an odd index and col is an even index are such cells.
  // 
  // Each cell of matrix A_beta_factors represent
  // all the directed paths starting at the vertex equal to the
  // column index of the matrix and ending at the vertex equal to
  // the row index of the matrix.
  // 
  // Each cell of matrix A_beta_factors is an object of Tran_Mat_Cell class.
  vector< vector< Tran_Mat_Cell * >> A_beta_factors;

  // A set of (row, column) numbers of the 2D matrix A_beta_factors where
  // the cell (row, column) depends on β factors.
  set< pair< int, int >> beta_dependent_cells;

  // Maps each β to all the transition matrix cells that are dependent on it.
  multimap< pair< int, int >, pair< int, int >> beta2cell;

  double delta_t = 1.0;

  vector< Eigen::MatrixXd > transition_matrix_collection;

  AnalysisGraph(DiGraph G) : graph(G){};


  /**
   * Finds all the simple paths starting at the start vertex and
   * ending at the end vertex.
   * Paths found are appended to the influenced_by data structure in the Node
   * Uses find_all_paths_between_util() as a helper to recursively find the paths
   */
  void find_all_paths_between(int start, int end) {
    // Mark all the vertices are not visited
    for_each( vertices(), [&](int v) { this->graph[v].visited = false; });

    // Create a vector of ints to store paths.
    vector< int > path;

    this->find_all_paths_between_util(start, end, path);
  }


  /**
   * Used by find_all_paths_between()
   * Recursively finds all the simple paths starting at the start vertex and
   * ending at the end vertex.
   * Paths found are appended to the influenced_by data structure in the Node
   */
  void
  find_all_paths_between_util(int start, int end, vector< int > &path) 
  {
    // Mark the current vertex visited
    this->graph[start].visited = true;

    // Add this vertex to the path
    path.push_back( start );

    // If current vertex is the destination vertex, then
    //   we have found one path. Append that to the end node
    if (start == end) 
    {
      // Add this path to the relevant transition matrix cell
      if( A_beta_factors[ path.back() ][ path[0] ] == nullptr )
      {
        this->A_beta_factors[ path.back() ][ path[0] ] = new Tran_Mat_Cell( path[0], path.back() );
      }

      this->A_beta_factors[ path.back() ][ path[0] ]->add_path( path );

      // This transition matrix cell is dependent upon Each β along this path.
      pair< int, int > this_cell = make_pair( path.back(), path[0] );

      beta_dependent_cells.insert( this_cell );

      for( int v = 0; v < path.size() - 1; v++ )
      {
        this->beta2cell.insert( make_pair( make_pair( path[v], path[v+1] ), this_cell ));
                                           //make_pair( path.back(), path[0])));
      }

    } 
    else 
    { 
      // Current vertex is not the destination
      // Process all the vertices adjacent to the current node
      for_each( successors(start), 
          [&](int v) 
          {
            if (!graph[v].visited) 
            {
              this->find_all_paths_between_util(v, end, path);
            }
      });
    }

    // Remove current vertex from the path and make it unvisited
    path.pop_back();
    graph[start].visited = false;
  };


public:
  /**
   * A method to construct an AnalysisGraph object given a JSON-serialized list
   * of INDRA statements.
   *
   * @param filename The path to the file containing the JSON-serialized INDRA
   * statements.
   */
  static AnalysisGraph from_json_file(string filename) {
    auto json_data = load_json(filename);

    DiGraph G;
    std::unordered_map<string, int> nameMap = {};
    int i = 0;
    for (auto stmt : json_data) {
      if (stmt["type"] == "Influence" and stmt["belief"] > 0.9) {
        auto subj = stmt["subj"]["concept"]["db_refs"]["UN"][0][0];
        auto obj = stmt["obj"]["concept"]["db_refs"]["UN"][0][0];
        if (!subj.is_null() and !obj.is_null()) {

          auto subj_str = subj.dump();
          auto obj_str = obj.dump();

          // Add the nodes to the graph if they are not in it already
          for (auto c : {subj_str, obj_str}) {
            if (nameMap.count(c) == 0) {
              nameMap[c] = i;
              auto v = boost::add_vertex(G);
              G[v].name = c;
              i++;
            }
          }

          // Add the edge to the graph if it is not in it already
          auto [e, exists] =
              boost::add_edge(nameMap[subj_str], nameMap[obj_str], G);
          for (auto evidence : stmt["evidence"]) {
            auto annotations = evidence["annotations"];
            auto subj_adjectives = annotations["subj_adjectives"];
            auto obj_adjectives = annotations["obj_adjectives"];
            auto subj_adjective =
                (!subj_adjectives.is_null() and subj_adjectives.size() > 0)
                    ? subj_adjectives[0]
                    : "None";
            auto obj_adjective =
                (obj_adjectives.size() > 0) ? obj_adjectives[0] : "None";
            auto subj_polarity = annotations["subj_polarity"];
            auto obj_polarity = annotations["obj_polarity"];
            G[e].causalFragments.push_back(CausalFragment{
                subj_adjective, obj_adjective, subj_polarity, obj_polarity});
          }
        }
      }
    }
    return AnalysisGraph(G);
  }


  auto add_node() { return boost::add_vertex(graph); }


  auto add_edge(int i, int j) { boost::add_edge(i, j, graph); }
  
  
  auto edges() { return boost::make_iterator_range(boost::edges(graph)); }


  auto predecessors(int i) {
    return boost::make_iterator_range(boost::inv_adjacent_vertices(i, graph));
  }


  auto out_edges(int i) {
    return boost::make_iterator_range(boost::out_edges(i, graph));
  }


  /*
  vector<std::pair<int, int>> simple_paths(int i, int j) {
    vector<std::pair<int, int>> paths = {};
    for (auto s : successors(i)) {
      paths.push_back(std::make_pair(i, s));
      for (auto e : simple_paths(s, j)) {
        paths.push_back(e);
      }
    }
    return paths;
  }
  */


  void construct_beta_pdfs() {
    double sigma_X = 1.0;
    double sigma_Y = 1.0;
    auto adjective_response_map = construct_adjective_response_map();
    vector<double> marginalized_responses;
    for (auto [adjective, responses] : adjective_response_map) {
      for (auto response : responses) {
        marginalized_responses.push_back(response);
      }
    }
    marginalized_responses =
        KDE(marginalized_responses).resample(default_n_samples);

    for (auto e : edges()) {
      vector<double> all_thetas = {};
      for (auto causalFragment : graph[e].causalFragments) {
        auto subj_adjective = causalFragment.subj_adjective;
        auto obj_adjective = causalFragment.obj_adjective;
        auto subj_responses =
            lmap([&](auto x) { return x * causalFragment.subj_polarity; },
                 get(adjective_response_map,
                     subj_adjective,
                     marginalized_responses));
        auto obj_responses = lmap(
            [&](auto x) { return x * causalFragment.obj_polarity; },
            get(adjective_response_map, obj_adjective, marginalized_responses));
        for (auto [x, y] : iter::product(subj_responses, obj_responses)) {
          all_thetas.push_back(atan2(sigma_Y * y, sigma_X * x));
        }
      }
      graph[e].kde = KDE(all_thetas);
    }
  }


  /*
   * Find all the simple paths between all the paris of nodes of the graph
   */
  void find_all_paths() {
    auto verts = vertices();

    // Allocate the 2D array that keeps track of the cells of the transition matrix
    // (A) that are dependent on βs.
    // This function can be called anytime after creating the CAG.
    this->allocate_A_beta_factors();

    // Clear the multimap that keeps track of cells in the transition matrix
    // that are dependent on each β.
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
    int num_verts = boost::num_vertices( graph );

    for( int row = 0; row < num_verts; ++row )
    {
      for( int col = 0; col < num_verts; ++col )
      {
        if( this->A_beta_factors[ row ][ col ] != nullptr )
        {
          this->A_beta_factors[ row ][ col ]->allocate_datastructures();
        }
      }
    }
  }


  /*
   * Prints the simple paths found between all pairs of nodes of the graph
   * Groupd according to the starting and ending vertex.
   * find_all_paths() should be called before this to populate the paths
   */
  void print_all_paths() {
    int num_verts = boost::num_vertices( graph );

    cout << "All the simple paths of:" << endl;

    for( int row = 0; row < num_verts; ++row )
    {
      for( int col = 0; col < num_verts; ++col )
      {
        if( this->A_beta_factors[ row ][ col ] != nullptr )
        {
          this->A_beta_factors[ row ][ col ]->print_paths();
        }
      }
    }
  }


  // Given an edge (source, target vertex ids - a β=∂target/∂source), 
  // print all the transition matrix cells that are dependent on it.
  void print_cells_affected_by_beta( int source, int target )
  {
    typedef multimap< pair< int, int >,  pair< int, int > >::iterator MMAPIterator;

    pair< int, int > beta = make_pair( source, target );

    pair<MMAPIterator, MMAPIterator> res = this->beta2cell.equal_range( beta );

    cout << endl << "Cells of A afected by beta_(" << source << ", " << target << ")" << endl;

    for( MMAPIterator it = res.first; it != res.second; it++ )
    {
      cout << "(" << it->second.first * 2 << ", " << it->second.second * 2 + 1 << ") ";
    }
    cout << endl;
  }


  /*
   ==========================================================================
   Sampling and inference
   ----------------------
  
   This section contains code for sampling and Bayesian inference.
   ==========================================================================
  */

  // Sample elements of the stochastic transition matrix from the 
  // prior distribution, based on gradable adjectives.
  const vector< Eigen::MatrixXd > & sample_from_prior() 
  {
    // Add probability distribution functions constructed from gradable
    // adjective data to the edges of the analysis graph data structure.
    this->construct_beta_pdfs();

    // Find all directed simple paths of the CAG
    this->find_all_paths();

    this->transition_matrix_collection.clear();

    int num_verts = boost::num_vertices( graph );
    cout << "Number of vertices: " << num_verts << endl;

    // A base transition matrix with the entries that does not change across samples.
    /*
     *          0  1  2  3  4  5 
     *  var_1 | 1 Δt             | 0
     *        | 0  1  0  0  0  0 | 1 ∂var_1 / ∂t
     *  var_2 |       1 Δt       | 2
     *        | 0  0  0  1  0  0 | 3
     *  var_3 |             1 Δt | 4
     *        | 0  0  0  0  0  1 | 5
     *
     *  Based on the directed simple paths in the CAG, some of the remaining 
     *  blank cells would be filled with β related values
     *  If we include second order derivatives to the model, there would be
     *  three rows for each variable and some of the off diagonal elements
     *  of odd indexed rows would be non zero.
     */
    Eigen::MatrixXd base_mat = Eigen::MatrixXd::Identity( num_verts * 2, num_verts * 2 );

    // Fill the Δts
    for( int vert = 0; vert < 2 * num_verts; vert += 2 )
    {
      base_mat( vert, vert + 1 ) = this->delta_t;
    }
    
    for( int samp_num = 0; samp_num < default_n_samples; samp_num++ )
    {
      // Get a copy of the base transition matrix
      Eigen::MatrixXd tran_mat( base_mat );

      // Update the β factor dependent cells of this matrix
      for( auto & [row, col] : beta_dependent_cells )
      {
        tran_mat( row * 2, col * 2 + 1 ) = this->A_beta_factors[ row ][ col ]->sample_from_prior( this->graph, samp_num );
      }

      this->transition_matrix_collection.push_back( tran_mat );
    } 

    return this->transition_matrix_collection;

    /*
    vector<std::pair<int, int>> node_pairs;

    // Get all length-2 permutations of nodes in the graph
    for (auto [i, j] : iter::product(vertices(), vertices())) {
      if (i != j) {
        node_pairs.push_back(std::make_pair(i, j));
      }
    }

    unordered_map<int, unordered_map<int, vector<std::pair<int, int>>>>
        simple_path_dict;
    for (auto [i, j] : node_pairs) {
      int cutoff = 4;
      int depth = 0;
      vector<std::pair<int, int>> paths;
    }
    */
  }


  auto print_nodes() {
    for_each(vertices(), [&](auto v) { cout << v << endl; });
  }
  
  
  auto print_edges() {
    for_each(edges(), [&](auto e) {
      cout << "(" << source(e, graph) << ", " << target(e, graph) << ")"
           << endl;
    });
  }

  
  auto to_dot() {
    write_graphviz(
        cout, graph, make_label_writer(boost::get(&Node::name, graph)));
  }
};
