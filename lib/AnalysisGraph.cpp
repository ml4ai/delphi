#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <sqlite3.h>
#include <utility>
#include <Eigen/Dense>
#include <pybind11/eigen.h>

#include "itertools.hpp"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/algorithm/string.hpp>

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
    /*
    for( vector< Tran_Mat_Cell * > row : this->A_beta_factors )
    {
      for( Tran_Mat_Cell * tmcp : row )
      {
        if( tmcp != nullptr )
        {
          delete tmcp;
        }
      }
      row.clear();
    }
    */
    this->A_beta_factors.clear();

    int num_verts = boost::num_vertices( graph );

    for( int vert = 0; vert < num_verts; ++vert )
    {
      //this->A_beta_factors.push_back( vector< Tran_Mat_Cell * >( num_verts ));
      this->A_beta_factors.push_back( vector< std::shared_ptr< Tran_Mat_Cell >>( num_verts ));
    }
  }


  void print_A_beta_factors()
  {
    int num_verts = boost::num_vertices( graph );

    for( int row = 0; row < num_verts; ++row )
    {
      for( int col = 0; col < num_verts; ++col )
      {
        cout << endl << "Printing cell: (" << row << ", " << col << ") " << endl;
        //if( this->A_beta_factors[ row ][ col ] != nullptr )
        if( this->A_beta_factors[ row ][ col ] )
        {
          this->A_beta_factors[ row ][ col ]->print_beta2product();
        }
      }
    }
  }


private:

  // Maps each concept name to the vertex id of the vertex that concept is
  // represented in the CAG
  std::unordered_map<string, int> name_to_vertex = {};
  
  // A_beta_factors is a 2D array (vector of vectors)
  // that keeps track of the β factors involved with
  // each cell of the transition matrix A.
  // 
  // Accordign to our current model, which uses variables and their partial
  // derivatives with respect to each other ( x --> y, βxy = ∂y/∂x ),
  // atmost half of the transition matrix cells are affected by βs. 
  // According to the way we organize the transition matrix, the cells A[row][col]
  // where row is an even index and col is an odd index are such cells.
  // 
  // Each cell of matrix A_beta_factors represent
  // all the directed paths starting at the vertex equal to the
  // column index of the matrix and ending at the vertex equal to
  // the row index of the matrix.
  // 
  // Each cell of matrix A_beta_factors is an object of Tran_Mat_Cell class.
  //vector< vector< Tran_Mat_Cell * >> A_beta_factors;
  //vector< vector< std::unique_ptr< Tran_Mat_Cell >>> A_beta_factors;
  vector< vector< std::shared_ptr< Tran_Mat_Cell >>> A_beta_factors;

  // A set of (row, column) numbers of the 2D matrix A_beta_factors where
  // the cell (row, column) depends on β factors.
  set< pair< int, int >> beta_dependent_cells;

  // Maps each β to all the transition matrix cells that are dependent on it.
  multimap< pair< int, int >, pair< int, int >> beta2cell;

  double t = 0.0;
  double delta_t = 1.0;
  vector< Eigen::VectorXd > s0;
  Eigen::VectorXd s0_original;
  Eigen::MatrixXd A_original;
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
  // This needs to be populated sometime before calling that function!!!
  // Access this as 
  // observed_state_sequence[ time step ][ vertex ][ indicator ]
  vector< vector< vector< double >>> observed_state_sequence; 


  vector< Eigen::MatrixXd > transition_matrix_collection;
  
  // Remember the old β and the edge where we perturbed the β.
  // We need this to revert the system to the previous state if the proposal
  // got rejected.
  // In the python implementation the variable original_value
  // was used for the same purpose.
  pair< boost::graph_traits< DiGraph >::edge_descriptor, double > previous_beta;

  // TODO: I am introducing this variable as a substitute for self.original_value
  // found in the python implementation to implement calculate_Δ_log_prior()
  //
  // Cells of the transition matrix that got chagned after perturbing a β
  // and their previous values.
  // A vector of triples - (row, column, previous value)
  //vector< tuple< int, int, double >> A_cells_changed;

  // To keep track whetehr the log_likelihood is initialized for the 1st time
  bool log_likelihood_initialized = false;

  double log_likelihood;

  AnalysisGraph( DiGraph G, std::unordered_map< string, int > name_to_vertex ) 
                : graph( G ),
                  name_to_vertex( name_to_vertex ) {};


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
    //   we have found one path.
    //   Add this cell to the Tran_Mat_Object that is tracking
    //   this transition matrix cell.
    if (start == end) 
    {
      // Add this path to the relevant transition matrix cell
      //if( A_beta_factors[ path.back() ][ path[0] ] == nullptr )
      if( ! A_beta_factors[ path.back() ][ path[0] ] )
      {
        //this->A_beta_factors[ path.back() ][ path[0] ] = new Tran_Mat_Cell( path[0], path.back() );
        this->A_beta_factors[ path.back() ][ path[0] ].reset( new Tran_Mat_Cell( path[0], path.back() ));
      }

      this->A_beta_factors[ path.back() ][ path[0] ]->add_path( path );

      // This transition matrix cell is dependent upon Each β along this path.
      pair< int, int > this_cell = make_pair( path.back(), path[0] );

      beta_dependent_cells.insert( this_cell );

      for( int v = 0; v < path.size() - 1; v++ )
      {
        this->beta2cell.insert( make_pair( make_pair( path[v], path[v+1] ), this_cell ));
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


  // Random number generator shared by all 
  // the random number generation tasks of AnalysisGraph
  // Note: If we are usign multiple threads, they should not share
  // one generator. We need to have a generator per thread
  std::mt19937 rand_num_generator;


  // Uniform distribution used by the MCMC sampler
  std::uniform_real_distribution< double > uni_dist;

  // Normal distrubution used to perturb β
  std::normal_distribution< double > norm_dist;

  void initialize_random_number_generator()
  {
    // Define the random number generator
    // All the places we need random numbers, share this generator
    std::random_device rd;
    this->rand_num_generator = std::mt19937( rd() );

    // Uniform distribution used by the MCMC sampler
    this->uni_dist = std::uniform_real_distribution< double >( 0.0, 1.0 );

    // Normal distrubution used to perturb β
    this->norm_dist = std::normal_distribution< double >( 0.0, 1.0);

    this->construct_beta_pdfs();
    this->find_all_paths();
  }


public:

  ~AnalysisGraph()
  {
      // Free memeroy allocated for Tran_Mat_Cell objects
      // that were used track β dependent cells in the transition matrix
      /*    
      for( auto & [row, col] : this->beta_dependent_cells )
      {
        //if( this->A_beta_factors[ row ][ col ] != nullptr )
        {
          cout << "Freeing (" << row << ", " << col << ")\n";
          cout << this->A_beta_factors[ row ][ col ].use_count()<< endl;
          //this->A_beta_factors[ row ][ col ]->print_paths();
          //delete this->A_beta_factors[ row ][ col ];
        }
      }
      */
  }


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

    for (auto stmt : json_data) {
      if (stmt["type"] == "Influence" and stmt["belief"] > 0.9) {
        auto subj = stmt["subj"]["concept"]["db_refs"]["UN"][0][0];
        auto obj = stmt["obj"]["concept"]["db_refs"]["UN"][0][0];
        if (!subj.is_null() and !obj.is_null()) {

          auto subj_str = subj.dump();
          auto obj_str = obj.dump();

          // Add the nodes to the graph if they are not in it already
          for ( string name : {subj_str, obj_str}) {
            if ( nameMap.find( name ) == nameMap.end()) {
              int v = boost::add_vertex( G );
              nameMap[ name ] = v;
              G[v].name = name;
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
            
            Event subject{ subj_adjective, subj_polarity, "" };
            Event object{ obj_adjective, obj_polarity, "" };
            G[e].evidence.push_back( Statement{ subject, object });
          }
        }
      }
    }
    AnalysisGraph ag =  AnalysisGraph( G, nameMap );
    ag.initialize_random_number_generator();
    return ag;
  }


  /**
   * A method to construct an AnalysisGraph object given from a vector of  
   * ( subject, object ) pairs (Statements)
   *
   * @param statements: A vector of Statement objects
   */
  //static AnalysisGraph from_statements( vector< Statement > statements )
  static AnalysisGraph from_statements( vector< pair< tuple< string, int, string >, tuple< string, int, string > >> statements )
  {
    DiGraph G;

    std::unordered_map<string, int> nameMap = {};

    //for ( Statement stmt : statements )
    for ( pair< tuple< string, int, string >, tuple< string, int, string > > stmt : statements )
    {
      Event subject = Event( stmt.first );
      Event object = Event( stmt.second );

      //Event subject = stmt.subject;
      //Event object = stmt.object;

      /*
      vector< string > concept;
      boost::split( concept, subject.concept_name, boost::is_any_of( "/" ));
      string subj_name = concept.back();

      concept.clear();
      boost::split( concept, object.concept_name, boost::is_any_of( "/" ));
      string obj_name = concept.back();
      */
      string subj_name = subject.concept_name;
      string obj_name = object.concept_name;

      // Add the nodes to the graph if they are not in it already
      for ( string name : {subj_name, obj_name} ) 
      {
        if ( nameMap.find( name ) == nameMap.end() )
        {
          int v = boost::add_vertex( G );
          nameMap[ name ] = v;
          G[v].name = name;
        }
      }

      // Add the edge to the graph if it is not in it already
      auto [e, exists] =
          boost::add_edge(nameMap[subj_name], nameMap[obj_name], G);

      //G[e].evidence.push_back( stmt );
      G[e].evidence.push_back( Statement{ subject, object } );
    }
    AnalysisGraph ag =  AnalysisGraph( G, nameMap );
    ag.initialize_random_number_generator();
    return ag;
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


  void set_initial_state()
  {
    this->A_original = sample_from_prior()[ 0 ];
    this->s0_original = this->construct_default_initial_state();
    
    for( boost::graph_traits< DiGraph >::edge_descriptor e : this->edges() )
    {
      int src = boost::source( e, this->graph );
      int tgt = boost::target( e, this->graph );

      // Initialize ∂source / ∂t values
      // *Loren: See the next comment from me below*
      this->s0_original( 2 * src + 1 ) = 0.1 * this->uni_dist( this->rand_num_generator );

      //this->A_original( 2 * tgt, 2 * src + 1 ) = 0.0;
    }

    // Given the initial latent state vector and the sampled transition matrix,
    // sample a sequence of latent states and observed states
    //
    // *Loren: So this isn't an absolutely necessary training step, really
    // sample_from_likelihood is just to generate predictions after the
    // training process, maybe we should rename it sample_predictions. Also
    // only the initial latent state needs to be initialized as specified in
    // the google doc, the rest of the latent_state_sequence is actually
    // learned from that initial latent state.* 
    this->sample_from_likelihood();
    this->latent_state_sequence = this->latent_state_sequences[ 0 ];
    this->observed_state_sequence = this->observed_state_sequences[ 0 ];
  }


  void take_step()
  {
    cout << "A before step:\n" << this->A_original << endl;

    this->A_original = this->sample_from_posterior( this->A_original );

    cout << "A after step:\n" << this->A_original << endl;
  }


  double get_beta( string source_vertex_name, string target_vertex_name )
  {
    // This is ∂target / ∂source
    return this->A_original( 2 * this->name_to_vertex[ target_vertex_name ], 
                             2 * this->name_to_vertex[ source_vertex_name ] + 1 );
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
      
      for ( Statement stmt : graph[e].evidence ) 
      {
        Event subject = stmt.subject;
        Event object = stmt.object;

        string subj_adjective = subject.adjective;
        string obj_adjective = object.adjective;
        
        auto subj_responses =
            lmap([&](auto x) { return x * subject.polarity; },
                 get(adjective_response_map,
                     subj_adjective,
                     marginalized_responses));

        auto obj_responses = lmap(
            [&](auto x) { return x * object.polarity; },
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
        //if( this->A_beta_factors[ row ][ col ] != nullptr )
        if( this->A_beta_factors[ row ][ col ] )
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
        //if( this->A_beta_factors[ row ][ col ] != nullptr )
        if( this->A_beta_factors[ row ][ col ] )
        {
          this->A_beta_factors[ row ][ col ]->print_paths();
        }
      }
    }
  }


  // Given an edge (source, target vertex ids - i.e. a β ≡ ∂target/∂source), 
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
    //this->construct_beta_pdfs();

    // Find all directed simple paths of the CAG
    //this->find_all_paths();

    this->transition_matrix_collection.clear();

    int num_verts = boost::num_vertices( this->graph );

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

            return gaussian( this->rand_num_generator ); 
          });
    }

    return observed_state;
  }


  // A method just to test sample_from_proposal( Eigen::MatrixXd A )
  // Just for debugging purposese
  void sample_from_proposal_debug()
  {
    int num_verts = boost::num_vertices( this->graph );

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero( 2 * num_verts, 2 * num_verts );
    // Update the β factor dependent cells of this matrix
    for( auto & [row, col] : this->beta_dependent_cells )
    {
      A( row * 2, col * 2 + 1 ) = this->A_beta_factors[ row ][ col ]->compute_cell( this->graph );
    }
    cout << endl << "Before Update: " << endl << A << endl;

    this->sample_from_proposal( A );

    cout << endl << "After Update: " << endl << A << endl;
  }


  /**
   * Find all the transition matrix (A) cells that are dependent on the β 
   * attached to the provided edge and update them.
   *
   * @param A: The current transitin matrix
   * @param e: The directed edge ≡ β that has been perturbed
   */
  void update_transition_matrix_cells( Eigen::MatrixXd & A, boost::graph_traits< DiGraph >::edge_descriptor e )
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

      A( row * 2, col * 2 + 1 ) = this->A_beta_factors[ row ][ col ]->compute_cell( this->graph );
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
   * @param A: Transition matrix
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
    std::sample( edge_it.begin(), edge_it.end(), e.begin(), 1, this->rand_num_generator ); 

    // Remember the previous β
    this->previous_beta = make_pair( e[0], this->graph[ e[0] ].beta );

    // Perturb the β
    // TODO: Check whether this perturbation is accurate
    graph[ e[0] ].beta += this->norm_dist( this->rand_num_generator );

    this->update_transition_matrix_cells( A, e[0] );
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


  // Now we are perturbing multiple cell of the transition matrix (A) that are
  // dependent on the randomly selected β (edge in the CAG).
  // In the python implementation, which is wrong, it randomly selects a single
  // cell in the transition matrix (A) and perturbs it. So, the python
  // implementation of this method is based on that incorrect pertubation.
  // 
  // Updated the logic of this function in the C++ implementation after having
  // a discussion with Adarsh.
  // TODO: Check with Adrash whether this new implementation is correct
  double calculate_delta_log_prior( Eigen::MatrixXd A )
  {
    // If kde of an edge is truely optional ≡ there are some
    // edges without a kde assigned, we should not access it
    // using .value() (In the case of kde being missing, this 
    // will throw an exception). We should follow a process
    // similar to Tran_Mat_Cell::sample_from_prior
    KDE & kde = this->graph[ this->previous_beta.first ].kde.value();

    // We have to return: log( p( β_new )) - log( p( β_old )) 
    return kde.logpdf( this->graph[ this->previous_beta.first ].beta ) - kde.logpdf( this->previous_beta.second );

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
    // log_likelihood just after sampling the initial transition matrix A,
    // make sure log_likelihood is initialize before calling this method
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

    if( acceptance_probability < uni_dist( this->rand_num_generator ))
    {
      // Reject the sample
      this->log_likelihood = original_log_likelihood;
 
      this->graph[ this->previous_beta.first ].beta = this->previous_beta.second;

      // Reset the transition matrix cells that were changed
      // TODO: Can we change the transition matrix only when the sample is accpeted?
      this->update_transition_matrix_cells( A, this->previous_beta.first );
    }

    return A;
  }


  /*
    ==========================================================================
    Basic Modeling Interface (BMI)
    ==========================================================================
  */
  //*Loren: So this section in AnalysisGraph.py is outdated and is not
  //currently used at all in evaluation.py. I think originally it was here to
  //control all modeling aspects such as the training and predictions, but has
  //been solely neglected. I found that it was basically easier to just work
  //around it than update this section. I think we can talk about revamping
  //this section, especially the config file aspect will be useful going
  //forward.* 

  /**
   * Create a BMI config file to initialize the model.
   *
   * @param filename: The filename with which the config file should be saved.
   */
  void create_bmi_config_file( string filename = "bmi_config.txt" )
  {
    Eigen::VectorXd s0 = this->construct_default_initial_state();

    std::ofstream file( filename.c_str() );

    file << s0;

    file.close();
  }


  /**
   * The default update function for a CAG node.
   *
   * @param vert_id: vertex id of the CAG node
   *
   * @return A vector of values corresponding to the distribution of the value of
   *         the real-valued variable representing the node.
   */ 
  vector< double > default_update_function( int vert_id )
  {
    vector< double > xs( default_n_samples );

    // Note: Both transition_matrix_collection and s0 should have
    //       default_n_samples elements.
    std::transform( this->transition_matrix_collection.begin(),
                    this->transition_matrix_collection.end(),
                    this->s0.begin(),
                    xs.begin(),
                    [&]( Eigen::MatrixXd A, Eigen::VectorXd s )
                    {
                      // The row of the transition matrix (A) that is associated
                      // with vertex vert_id = 2 * vert_id
                      return A.row( 2 * vert_id ) * s;
                    });

    return xs;
  }


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

                return gaussian( this->rand_num_generator ); 
              });
        }
      }
    }
  }


  /*
    ==========================================================================
    Model parameterization
    *Loren: I am going to try to port this, I'll try not to touch anything up top
    and only push changes that compile. If I do push a change that breaks things
    you could probably just comment out this section.*
    ==========================================================================
  */


  /** 
   * Map each concept node in the AnalysisGraph instance to one or more
   * tangible quantities, known as 'indicators'.
   *
   * @param n: Int representing number of indicators to attach per node.
   * Default is 1 since our model so far is configured for only 1 indicator per
   * node.
   */ 
  void map_concepts_to_indicators( int n=1 )
  {
    sqlite3 *db;
    int rc = sqlite3_open(std::getenv("DELPHI_DB"), &db);
    if (!rc)
      print("Opened db successfully");
    else
      print("Could not open db");

    sqlite3_stmt *stmt;
    string query_base = "select Source, Indicator from concept_to_indicator_mapping ";
    string query;
    for( int v : this->vertices() )
    {
      query = query_base + "where `Concept` like " + "'" + this->graph[ v ].name +"'";
      rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
      this->graph[ v ].indicators.clear();
      this->graph[ v ].indicator_names.clear();
      for( int c = 0; c < n; c = c + 1 ) 
      { 
        rc = sqlite3_step(stmt);
        if (rc == SQLITE_ROW)
        {
          string ind_source = std::string(
              reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
          string ind_name = std::string(
              reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1)));
          Indicator ind = Indicator(ind_name,ind_source);
          this->graph[ v ].indicators.push_back(ind);
          this->graph[ v ].indicator_names[ind_name] = c;
        } else {
          cout << "No more data, only " << c << "indicators attached to " << this->graph[ v ].name << endl;
        }
      }
      sqlite3_finalize(stmt);
    }
    sqlite3_close(db);
  }


  auto print_nodes() {
    cout << "Vertex IDs and their names in the CAG" << endl;
    cout << "Vertex ID : Name" << endl;
    cout << "--------- : ----" << endl;
    for_each(vertices(), [&](auto v) { cout << v << "         : " << this->graph[ v ].name << endl; });
  }
  
  
  auto print_edges() {
    for_each(edges(), [&](auto e) {
      cout << "(" << source(e, graph) << ", " << target(e, graph) << ")"
           << endl;
    });
  }


  void print_name_to_vertex()
  {
    for( auto [ name, vert ] : this->name_to_vertex )
    {
      cout << name << " -> " << vert << endl;
    }
    cout << endl;
  }
  

  auto to_dot() {
    write_graphviz(
        cout, graph, make_label_writer(boost::get(&Node::name, graph)));
  }


  auto print_indicators()
  {
    for( int v : this->vertices() )
    {
      cout << "node " << v << ": " << this->graph[ v ].name << ":" << endl;
      for( auto [ name, vert ] : this->graph[ v ].indicator_names )
      {
        cout << "\t" << "indicator " << vert << ": " << name << endl;
      }

    }
  }
};
