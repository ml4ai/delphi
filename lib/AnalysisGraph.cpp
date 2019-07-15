#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <sqlite3.h>
#include <utility>
#include <Eigen/Dense>
#include <pybind11/eigen.h>

#include "../external/cppitertools/itertools.hpp"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>

#include "AnalysisGraph.hpp"
#include "utils.hpp"
#include "tran_mat_cell.hpp"

#include <typeinfo>

using std::cout, std::endl, std::unordered_map, std::pair, std::string, std::ifstream,
    std::stringstream, std::map, std::multimap, std::make_pair, std::tuple, std::make_tuple,
    std::set, boost::inner_product, boost::edge, boost::source, boost::target,
    boost::graph_bundle, boost::make_label_writer, boost::write_graphviz,
    boost::lambda::make_const, utils::load_json, utils::hasKey, utils::get, utils::lmap;


const size_t default_n_samples = 100;


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
  AnalysisGraph() {}
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

    //cout << A_beta_factors.size() << endl;
    //cout << A_beta_factors[0].size() << endl;
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
  // TODO: Need to free these pointers in the destructor.
  vector< vector< Tran_Mat_Cell * >> A_beta_factors;

  // A set of (row, column) numbers of the 2D matrix A_beta_factors where
  // the cell (row, column) depends on β factors.
  set< pair< int, int >> beta_dependent_cells;

  // Maps each β to all the transition matrix cells that are dependent on it.
  multimap< pair< int, int >, pair< int, int >> beta2cell;

  double t = 0.0;
  double delta_t = 1.0;
  vector< Eigen::VectorXd > s0;
  Eigen::VectorXd s0_original;
  int n_timesteps;

  // Access this as
  // latent_state_sequences[ sample ][ time step ]
  vector< vector< Eigen::VectorXd >> latent_state_sequences; 

  // Access this as
  // latent_state_sequence[ time step ]
  vector< Eigen::VectorXd > latent_state_sequence; 

  // Access this as 
  // observed_state_sequences[ sample ][ time step ][ vertex ][ indicator ]
  vector< vector< vector< vector< double >>>> observed_state_sequences; 

  // TODO: In the python file I cannot find the place where 
  // self.observed_state_sequence gets populated.
  // It only appears within the calculate_log_likelihood and
  // there it is beign accessed!!!
  // I am defining this here so taht I can complete the implementation of 
  // calculate_log_likelihood().
  // This needs to be populated somtime before calling that function!!!
  // Access this as 
  // observed_state_sequence[ time step ][ vertex ][ indicator ]
  vector< vector< vector< double >>> observed_state_sequence; 


  vector< Eigen::MatrixXd > transition_matrix_collection;
  
  // Remember the ratio: β_old / β_new and the edge where we perturbed the β.
  // We need this to revert the system to the previous state if the proposal
  // got rejected. To revert, we have to:
  // graph[ beta_revert_ratio.first ] *= beta_revert_ratio.second;
  // and update the cells of A that are dependent on this β with
  // update_cell( make_pair( source(brr.first, graph), target(brr.first, graph) ), brr.second )
  // In the python implementation the variable original_value
  // was used for the same purpose.
  pair< boost::graph_traits< DiGraph >::edge_descriptor, double > beta_revert_ratio;

  // TODO: I am introducing this variable as a sustitute for self.original_value
  // found in the python implementation to implement calculate_Δ_log_prior()
  //
  // Cells of the transition matrix that got chagned after perturbing a β
  // and their previous values.
  // A vector of triples - (row, column, previous value)
  //vector< tuple< int, int, double >> A_cells_changed;

  // To keep track whetehr the log_likelihood is initialized for the 1st time
  bool log_likelihood_initialized = false;

  double log_likelihood;

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


  /*
   ==========================================================================
   Utilities
   ==========================================================================
  */
  Eigen::VectorXd construct_default_initial_state( )
  { 
    // Let vertices of the CAG be v = 0, 1, 2, 3, ...
    // Then,
    //    indexes 2*v keeps track of the state of each variable v
    //    indexes 2*v+1 keeps track of the state of ∂v/∂t
    int num_verts = boost::num_vertices( graph );
    int num_els = num_verts * 2;

    Eigen::VectorXd init_st( num_els );
    init_st.setZero();

    for( int i = 0; i < num_els; i += 2 )
    {
      init_st( i ) = 1.0;
    }

    return init_st;
  }


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
      // TODO: Why kde is optional in struct Edge?
      // It seems all the edges get assigned with a kde
      graph[e].kde = KDE(all_thetas);

      // Initialize the initial β for this edge
      // TODO: Decide the correct way to initialize this
      graph[ e ].beta = graph[ e ].kde.value().mu;
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

  // TODO: Need testing
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

    int num_verts = boost::num_vertices( this->graph );
    //cout << "Number of vertices: " << num_verts << endl;

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
      for( auto & [row, col] : this->beta_dependent_cells )
      {
        tran_mat( row * 2, col * 2 + 1 ) = this->A_beta_factors[ row ][ col ]->sample_from_prior( this->graph, samp_num );
      }

      this->transition_matrix_collection.push_back( tran_mat );
    } 

    return this->transition_matrix_collection;
  }


  // TODO: Need testing
  /**
   * Sample a collection of observed state sequences from the likelihood
   * model given a collection of transition matrices.
   *
   * @param n_timesteps: The number of timesteps for the sequences.
   */
  void sample_from_likelihood( int n_timesteps=10 )
  {
    this->n_timesteps = n_timesteps;

    int num_verts = boost::num_vertices( this->graph );

    // Allocate memory for latent_state_sequences
    this->latent_state_sequences = vector< vector< Eigen::VectorXd >>( default_n_samples, 
                                           vector< Eigen::VectorXd > ( n_timesteps, 
                                                   Eigen::VectorXd   ( num_verts * 2 )));

    for( int samp = 0; samp < default_n_samples; samp++ )
    {
      this->latent_state_sequences[ samp ][ 0 ] = this->s0_original;

      for( int ts = 1; ts < n_timesteps; ts++ )
      {
        //cout << this->transition_matrix_collection[ samp ] << endl; 
        //cout << this->latent_state_sequences[ samp ][ ts-1 ] << endl; 
        this->latent_state_sequences[ samp ][ ts ] = this->transition_matrix_collection[ samp ] * this->latent_state_sequences[ samp ][ ts-1 ];
      }
    }

    // Allocate memory for observed_state_sequences
    this->observed_state_sequences = vector< vector< vector< vector< double >>>>( default_n_samples, 
                                             vector< vector< vector< double >>> ( n_timesteps,
                                                     vector< vector< double >>  ( )));
    
    for( int samp = 0; samp < default_n_samples; samp++ )
    {
      vector< Eigen::VectorXd > & sample = this->latent_state_sequences[ samp ];

      std::transform( sample.begin(), sample.end(), this->observed_state_sequences[ samp ].begin(), [this]( Eigen::VectorXd latent_state ){ return this->sample_observed_state( latent_state ); });
    }
  }


  // TODO: Need testing
  /**
   * Sample observed state vector.
   * This is the implementation of the emission function.
   * 
   * @param latent_state: Latent state vector.
   *                      This has 2 * number of vertices in the CAG.
   *                      Even indices track the state of each vertex.
   *                      Odd indices track the state of the derivative.
   *
   * @return Observed state vector. Observed state for each indicator for each vertex.
   *         Indexed by: [ vertex id ][ indicator id ]
   */
  vector< vector< double >> sample_observed_state( Eigen::VectorXd latent_state )
  {
    int num_verts = boost::num_vertices( this->graph );
    
    assert( num_verts == latent_state.size() / 2 );

    vector< vector< double >> observed_state( num_verts );

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator( seed );

    for( int v = 0; v < num_verts; v++ )
    {
      vector< Indicator > & indicators = this->graph[ v ].indicators;

      observed_state[ v ] = vector< double >( indicators.size() );

      // Sample observed value of each indicator around the mean of the indicator
      // scaled by the value of the latent state that caused this observation.
      // TODO: Question - Is ind.mean * latent_state[ 2*v ] correct?
      //                  Shouldn't it be ind.mean + latent_state[ 2*v ]?
      std::transform( indicators.begin(), indicators.end(), observed_state[ v ].begin(), 
          [&]( Indicator ind )
          {
            std::normal_distribution< double > gaussian( ind.mean * latent_state[ 2*v ], ind.stdev );

            return gaussian( generator ); 
          });
    }

    return observed_state;
  }


  // A method just to test sample_from_proposal( Eigen::MatrixXd A )
  void sample_from_proposal_debug()
  {
    // Just for debugging purposese
    int num_verts = boost::num_vertices( this->graph );

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero( 2 * num_verts, 2 * num_verts );
    // Update the β factor dependent cells of this matrix
    for( auto & [row, col] : this->beta_dependent_cells )
    {
      A( row * 2, col * 2 + 1 ) = this->A_beta_factors[ row ][ col ]->compute_cell( );
    }
    cout << endl << "Before Update: " << endl << A << endl;

    this->sample_from_proposal( A );

    cout << endl << "After Update: " << endl << A << endl;
  }


  /**
   * Find all the transition matrix (A) cells that are dependent on this β 
   * and update them.
   */
  void update_transition_matrix_cells( Eigen::MatrixXd & A, boost::graph_traits< DiGraph >::edge_descriptor e, double beta_ratio  )
  {
    pair< int, int > beta = make_pair( boost::source( e, this->graph ), boost::target( e, this->graph ) ); 

    typedef multimap< pair< int, int >,  pair< int, int > >::iterator MMAPIterator;

    pair<MMAPIterator, MMAPIterator> res = this->beta2cell.equal_range( beta );

    // TODO: I am introducing this to implement calculate_Δ_log_prior
    // Remember the cells of A that got changed and their previous values
    //this->A_cells_changed.clear();

    for( MMAPIterator it = res.first; it != res.second; it++ )
    {
      int & row = it->second.first;
      int & col = it->second.second;;

      // Note that I am remembering row and col instead of 2*row and 2*col+1
      // row and col resembles an edge in the CAG: row -> col
      // ( 2*row, 2*col+1 ) is the transition mateix cell that got changed.
      //this->A_cells_changed.push_back( make_tuple( row, col, A( row * 2, col * 2 + 1 )));

      A( row * 2, col * 2 + 1 ) = this->A_beta_factors[ row ][ col ]->update_cell( beta, beta_ratio );
    }
  }
  

  /**
   * Sample a new transition matrix from the proposal distribution,
   * given a current candidate transition matrix.
   * In practice, this amounts to:
   *    Selecting a random β.
   *    Perturbing it a bit.
   *    Updating all the transition matrix cells that are dependent on it.
   * 
   * @parm A: Transition matrix
   */
  // TODO: Need testng
  // TODO: Before calling sample_from_proposal() we must call 
  // AnalysisGraph::find_all_paths()
  // TODO: Before calling sample_from_proposal(), we mush assign initial βs and
  // run Tran_Mat_Cell::compute_cell() to initialize the first transistion matrix.
  // TODO: Update Tran_Mat_Cell::compute_cell() to calculate the proper value.
  // At the moment it just computes sum of length of all the paths realted to this cell
  void sample_from_proposal( Eigen::MatrixXd & A )
  {
    // Randomly pick an edge ≡ β 
    boost::iterator_range edge_it = this->edges();

    vector< boost::graph_traits< DiGraph >::edge_descriptor > e(1); 
    std::sample( edge_it.begin(), edge_it.end(), e.begin(), 1, std::mt19937{ std::random_device{}() }); 

    //cout << source( e[0], graph ) << ", " << target( e[0], graph ) << endl;

    // Remember the previous β
    double prev_beta = this->graph[ e[0] ].beta;

    // Perturb the β
    // TODO: Check whether this perturbation is accurate
    graph[ e[0] ].beta += sample_from_normal( 0.0, 0.01 ); // Defined in kde.hpp

    double beta_ratio = this->graph[ e[0] ].beta / prev_beta;

    this->beta_revert_ratio = make_pair( e[0], 1.0 / beta_ratio );

    this->update_transition_matrix_cells( A, e[0], beta_ratio );

    /*
    // Find all the transition matrix (A) cells that are dependent on this β 
    // and update them.
    pair< int, int > beta = make_pair( boost::source( e[0], this->graph ), boost::target( e[0], this->graph ) ); 

    typedef multimap< pair< int, int >,  pair< int, int > >::iterator MMAPIterator;

    pair<MMAPIterator, MMAPIterator> res = this->beta2cell.equal_range( beta );

    // TODO: I am introducing this to implement calculate_Δ_log_prior
    // Remember the cells of A that got changed and their previous values
    //this->A_cells_changed.clear();

    for( MMAPIterator it = res.first; it != res.second; it++ )
    {
      int & row = it->second.first;
      int & col = it->second.second;;

      // Note that I am remembering row and col instead of 2*row and 2*col+1
      // row and col resembles an edge in the CAG: row -> col
      // ( 2*row, 2*col+1 ) is the transition mateix cell that got changed.
      //this->A_cells_changed.push_back( make_tuple( row, col, A( row * 2, col * 2 + 1 )));

      A( row * 2, col * 2 + 1 ) = this->A_beta_factors[ row ][ col ]->update_cell( beta, beta_ratio );
    }
    */
  }


  void set_latent_state_sequence( Eigen::MatrixXd A )
  {
    int num_verts = boost::num_vertices( this->graph );

    // Allocate memory for latent_state_sequence
    this->latent_state_sequence = vector< Eigen::VectorXd > ( this->n_timesteps, 
                                          Eigen::VectorXd   ( num_verts * 2 ));

    // TODO: Disambiguate the type of s0 in the python implementation of this method.
    // in the AnalysisGraph::initialize method, the python implementation sets
    // s0 to a list of pandas Series.
    // I'm not sure whether that is what we need here. I'm using s0_original, 
    // one element of s0 here for the moment.
    this->latent_state_sequence[ 0 ] = this->s0_original;

    for( int ts = 1; ts < this->n_timesteps; ts++ )
    {
      this->latent_state_sequence[ ts ] = A * this->latent_state_sequence[ ts-1 ];
    }
  }


  double log_normpdf( double x, double mean, double sd )
  {
    double var = pow( sd, 2 );
    double log_denom = -0.5 * log( 2 * M_PI ) - log( sd );
    double log_nume = pow( x - mean, 2 ) / ( 2 * var );

    return log_denom - log_nume;
  }


  double calculate_log_likelihood( Eigen::MatrixXd A )
  {
    double log_likelihood_total = 0.0;

    this->set_latent_state_sequence( A );
    
    int num_ltsts = this->latent_state_sequence.size();
    int num_obsts = this->observed_state_sequence.size();

    // TODO: Is it always the case that
    // num_timesteps = num_ltsts = num_obsts = this->n_timesteps?
    int num_timesteps = num_ltsts < num_obsts ? num_ltsts : num_obsts;

    for( int ts = 0; ts < num_timesteps; ts++ )
    {
      Eigen::VectorXd latent_state = this->latent_state_sequence[ ts ];

      // TODO: In the python file I cannot find the place where 
      // self.observed_state_sequence gets populated.
      // It only appears within the calculate_log_likelihood and
      // there it is beign accessed!!!
      // Access
      // observed_state[ vertex ][ indicator ]
      vector< vector< double >> observed_state = this->observed_state_sequence[ ts ];

      for( int v : vertices() )
      {
        int num_inds_for_v = observed_state[ v ].size();

        for( int indicator = 0; indicator < num_inds_for_v; indicator++ )
        {
          double value = observed_state[ v ]           [ indicator ];
          Indicator ind =         graph[ v ].indicators[ indicator ];

          // Even indices of latent_state keeps track of the state of each vertex
          double log_likelihood = this->log_normpdf( value, latent_state[ 2 * v ] * ind.mean, ind.stdev );
          log_likelihood_total += log_likelihood; 
        }
      }
    }

    return log_likelihood_total;
  }


  // TODO: This implementation is WRONG!!!
  // Now we are perturbing multiple cell of the transition matrix (A) that are
  // dependent on the randomly selected β (edge in the CAG).
  // In the python implementation, which is wrong, it randomly selects a single
  // cell in the transition matrix (A) and perturb it. So, it defines the
  // current value and previous value of that cell.
  // TODO: We need to go back to Math and workout the Math before implementing
  // this method.
  // I have implemented a stub as a placeholder so taht I can move forward with
  // the AnalysisGraph::sample_from_posterior() method.
  double calculate_delta_log_prior( Eigen::MatrixXd A )
  {
    //this->beta_revert_ratio = make_pair( e[0], 1.0 / beta_ratio );
    // If kde of an edge is truely optional ≡ there are some
    // edges without a kde assigned, we should not access it
    // using .value() (In the case of kde being missing, this 
    // with throw and exception). We should follow a process
    // similar to Tran_Mat_Cell::sample_from_prior
    KDE & kde = this->graph[ this->beta_revert_ratio.first ].kde.value();

    const int & source = boost::source( this->beta_revert_ratio.first, this->graph );
    const int & target = boost::target( this->beta_revert_ratio.first, this->graph ); 

    // TODO: Now since we are changing multiple cells of A defining the previous
    // value is not stratight forward.
    return kde.logpdf( A( 2 * source, 2 * target + 1 ) / this->delta_t ); // - kde.logpdf( prev_val / this->delta_t );

    /*
    vector< double > log_diffs( this->A_cells_changed.size() );

    std::transform( this->A_cells_changed.begin(), this->A_cells_changed.end(),
                    log_diffs.begin(),
                    [&]( tuple< int, int, double > cell )
                    {
                      int & source = std::get< 0 >( cell );
                      int & target = std::get< 1 >( cell );
                      double & prev_val = std::get< 2 >( cell );

                      const pair< boost::graph_traits< DiGraph >::edge_descriptor, bool > & e = boost::edge( source, target, this->graph); 
                      
                      // If kde of an edge is truely optional ≡ there are some
                      // edges without a kde assigned, we should not access it
                      // using .value() (In the case of kde being missing, this 
                      // with throw and exception). We should follow a process
                      // similar to Tran_Mat_Cell::sample_from_prior
                      KDE & kde = this->graph[ e.first ].kde.value();

                      return kde.logpdf( A( 2 * source, 2 * target + 1 ) / this->delta_t ) - kde.logpdf( prev_val / this->delta_t );
                    });
    return std::accumulate( log_diffs.begin(), log_diffs.end(), 0.0 );
    */
  }


  /**
   * Run Bayesian inference - sample from the posterior distribution.
   */
  Eigen::MatrixXd sample_from_posterior( Eigen::MatrixXd A )
  {
    // TODO: This check will be called for each sample and will be false
    // except for the 1st time. It is better to remove this and initialize
    // log_likelihood just after sampling the initial transition matrix A
    // , make sure log_likelihood is initialize before calling this method
    // and get rid of this check from here.
    if( ! this->log_likelihood_initialized )
    {
      this->log_likelihood = this->calculate_log_likelihood( A );
      this->log_likelihood_initialized = true;
    }

    // Sample a new transition matrix from the proposal distribution
    this->sample_from_proposal( A );

    // TODO: AnalysisGraph::calculate_delat_log_prior() method is not properly
    // implemented. Only a stub is implemented. 
    double delta_log_prior = this->calculate_delta_log_prior( A );

    double original_log_likelihood = this->log_likelihood;
    double candidate_log_likelihood = this->calculate_log_likelihood( A );
    double delta_log_likelihood = candidate_log_likelihood - original_log_likelihood;

    double delta_log_joint_probability = delta_log_prior + delta_log_likelihood;

    double acceptance_probability = std::min( 1.0, exp( delta_log_joint_probability ));

    // Define the random number generator
    // TODO: We have to do this once before calling sample_from_posterior()
    // We have to move this out of sample_from_posterior()
    // Maybe we can define this in the class constructor and use it in
    // all the places we need random numbers.
    std::random_device rd;
    std::mt19937 mt( rd() );
    std::uniform_real_distribution< double > dist( 0.0, 1.0 );

    if( acceptance_probability < dist( mt ))
    {
      // Reject the sample
      this->log_likelihood = original_log_likelihood;
    
      // Reset the transition matrix cells that were changed
      // TODO: Can we change the transition matrix only when the sample is accpeted?
      this->update_transition_matrix_cells( A, this->beta_revert_ratio.first, this->beta_revert_ratio.second );
    }

    return A;
  }


  /*
    ==========================================================================
    Basic Modeling Interface (BMI)
    ==========================================================================
  */

  /**
   * Initialize the executable AnalysisGraph with a config file.
   * 
   * @param initialize_indicators: Boolean flag that sets whether indicators
   * are initialized as well.
   */
  void initialize( bool initialize_indicators=true )
  {
    this->t = 0.0;

    // Create a 'reference copy' of the initial latent state vector
    //    indexes 2*v keeps track of the state of each variable v
    //    indexes 2*v+1 keeps track of the state of ∂v/∂t
    this->s0_original = this->construct_default_initial_state();

    // Create default_n_samples copies of the initial latent state vector
    this->s0 = vector( default_n_samples, this->s0_original );

    for( int v : this->vertices() )
    {
      this->graph[ v ].rv.name = this->graph[ v ].name;
      this->graph[ v ].rv.dataset = vector< double >( default_n_samples, 1.0 );
      this->graph[ v ].rv.partial_t = this->s0_original[ 2 * v + 1 ];
      // TODO: Need to add the update_function variable to the node
      // n[1]["update_function"] = self.default_update_function
      //graph[ v ].update_function = ????;
      
      if( initialize_indicators )
      {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator( seed );

        for( Indicator ind : graph[ v ].indicators )
        {
          vector< double > & dataset = this->graph[ v ].rv.dataset;

          ind.samples.clear();
          ind.samples = vector< double >( dataset.size() );

          // Sample observed value of each indicator around the mean of the indicator
          // scaled by the value of the latent state that caused this observation.
          // TODO: Question - Is ind.mean * rv_datun correct?
          //                  Shouldn't it be ind.mean + rv_datum?
          std::transform( dataset.begin(), dataset.end(), ind.samples.begin(), 
              [&](int rv_datum)
              {
                std::normal_distribution< double > gaussian( ind.mean * rv_datum, 0.01 );

                return gaussian( generator ); 
              });
        }
      }
    }
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
