#pragma once

#include <Eigen/Dense>

#include <boost/graph/graph_traits.hpp>
#include <boost/range/iterator_range.hpp>

#include "graphviz_interface.hpp"

#include "DiGraph.hpp"
#include "tran_mat_cell.hpp"
#include <fmt/format.h>

const size_t DEFAULT_N_SAMPLES = 100;

enum InitialBeta { ZERO, ONE, HALF, MEAN, RANDOM };

typedef std::unordered_map<std::string, std::vector<double>>
    AdjectiveResponseMap;

typedef std::vector<std::vector<std::vector<double>>> ObservedStateSequence;

typedef std::pair<std::tuple<std::string, int, std::string>,
                  std::tuple<std::string, int, std::string>>
    CausalFragment;

AdjectiveResponseMap construct_adjective_response_map(size_t n_kernels);

/**
 * The AnalysisGraph class is the main model/interface for Delphi.
 */
class AnalysisGraph {
  DiGraph graph;

  public:
  AnalysisGraph() {}
  Node &operator[] (std::string);
  Edge& edge(int, int);

  // Manujinda: I had to move this up since I am usign this within the private:
  // block This is ugly. We need to re-factor the code to make it pretty again
  auto vertices();

  NEIGHBOR_ITERATOR successors(int i);

  // Allocate a num_verts x num_verts 2D array (std::vector of std::vectors)
  void allocate_A_beta_factors();

  void print_A_beta_factors();

  private:

  Node &operator[] (int);
  void clear_state(); 

  // Maps each concept name to the vertex id of the
  // vertex that concept is represented in the CAG
  // concept name --> CAV vertex id
  std::unordered_map<std::string, int> name_to_vertex = {};

  // Keeps track of indicators in CAG to ensure there are no duplicates.
  // std::vector<std::string> indicators_in_CAG;
  std::unordered_set<std::string> indicators_in_CAG;

  // A_beta_factors is a 2D array (std::vector of std::vectors) that keeps track
  // of the β factors involved with each cell of the transition matrix A.
  //
  // Accordign to our current model, which uses variables and their partial
  // derivatives with respect to each other ( x --> y, βxy = ∂y/∂x ),
  // atmost half of the transition matrix cells can be affected by βs.
  // According to the way we organize the transition matrix, the cells
  // A[row][col] where row is an even index and col is an odd index
  // are such cells.
  //
  // Each cell of matrix A_beta_factors represent all the directed paths
  // starting at the vertex equal to the column index of the matrix and
  // ending at the vertex equal to the row index of the matrix.
  //
  // Each cell of matrix A_beta_factors is an object of Tran_Mat_Cell class.
  std::vector<std::vector<std::shared_ptr<Tran_Mat_Cell>>> A_beta_factors;

  // A set of (row, column) numbers of the 2D matrix A_beta_factors
  // where the cell (row, column) depends on β factors.
  std::set<std::pair<int, int>> beta_dependent_cells;

  // Maps each β to all the transition matrix cells that are dependent on it.
  std::multimap<std::pair<int, int>, std::pair<int, int>> beta2cell;

  double t = 0.0;
  double delta_t = 1.0;
  std::vector<Eigen::VectorXd> s0;

  // Latent state that is evolved by sampling.
  // Since s0 is used to represent a sequence of latent states,
  // I named this s0_original. Once things are refactored, we might be able to
  // convert this to s0
  Eigen::VectorXd s0_original;

  // Transition matrix that is evolved by sampling.
  // Since variable A has been already used locally in other methods,
  // I chose to name this A_orginal. After refactoring the code, we could
  // rename this to A.
  Eigen::MatrixXd A_original;

  int n_timesteps;
  int pred_timesteps;
  std::pair<std::pair<int,int>,std::pair<int,int>> training_range;

  // Accumulates the transition matrices for accepted samples
  // Access: [ sample number ]
  // std::vector<Eigen::MatrixXd> training_sampled_transition_matrix_sequence;

  // Accumulates the latent states for accepted samples
  // Access this as
  // latent_state_sequences[ sample ][ time step ]
  std::vector<std::vector<Eigen::VectorXd>> training_latent_state_sequence_s;

  // This is a column of the
  // this->training_latent_state_sequence_s
  // prediction_initial_latent_state_s.size() = this->res
  // TODO: If we make the code using this variable to directly fetch the values
  // from this->training_latent_state_sequence_s, we can get rid of this
  std::vector<Eigen::VectorXd> prediction_initial_latent_state_s;
  std::vector<std::string> pred_range;

  // Access this as
  // prediction_latent_state_sequence_s[ sample ][ time step ]
  std::vector<std::vector<Eigen::VectorXd>> predicted_latent_state_sequence_s;

  // Access this as
  // prediction_observed_state_sequence_s
  //                            [ sample ][ time step ][ vertex ][ indicator ]
  std::vector<ObservedStateSequence> predicted_observed_state_sequence_s;

  // Sampling resolution. Default is 200
  int res = 200;

  // Training start
  int init_training_year;
  int init_training_month;

  // Keep track whether the model is trained.
  // Used to check whether there is a trained model before calling
  // generate_prediction()
  bool trained = false;

  // Access this as
  // latent_state_sequences[ sample ][ time step ]
  std::vector<std::vector<Eigen::VectorXd>> latent_state_sequences;

  // Access this as
  // latent_state_sequence[ time step ]
  std::vector<Eigen::VectorXd> latent_state_sequence;

  // Access this as
  // observed_state_sequences[ sample ][ time step ][ vertex ][ indicator ]
  std::vector<ObservedStateSequence> observed_state_sequences;

  // Access this as
  // observed_state_sequence[ time step ][ vertex ][ indicator ]
  ObservedStateSequence observed_state_sequence;

  std::vector<Eigen::MatrixXd> transition_matrix_collection;

  // Remember the old β and the edge where we perturbed the β.
  // We need this to revert the system to the previous state if the proposal
  // gets rejected.
  std::pair<boost::graph_traits<DiGraph>::edge_descriptor, double>
      previous_beta;

  double log_likelihood = 0.0;
  double previous_log_likelihood = 0.0;

  AnalysisGraph(DiGraph G, std::unordered_map<std::string, int> name_to_vertex)
      : graph(G), name_to_vertex(name_to_vertex){};

  void
  get_subgraph_rooted_at(int vert,
                         std::unordered_set<int>& vertices_to_keep,
                         int cutoff,
                         NEIGHBOR_ITERATOR (AnalysisGraph::*neighbors)(int));

  void get_subgraph_sinked_at(int vert,
                              std::unordered_set<int>& vertices_to_keep,
                              int cutoff);

  void get_subgraph_between(int start,
                            int end,
                            std::vector<int>& path,
                            std::unordered_set<int>& vertices_to_keep,
                            int cutoff);

  /**
   * Finds all the simple paths starting at the start vertex and
   * ending at the end vertex.
   * Uses find_all_paths_between_util() as a helper to recursively find the
   * paths
   */
  void find_all_paths_between(int start, int end, int cutoff);

  /**
   * Recursively finds all the simple paths starting at the start vertex and
   * ending at the end vertex. Used by find_all_paths_between()
   * Paths found are added to the Tran_Mat_Cell object that is tracking the
   * transition matrix cell (2*end, 2*start)
   *
   * @param start: Start vertex of the path
   * @param end  : End vertex of the path
   * @param path : A path starting at vettex start that is being explored
   *
   * @return void
   */
  void find_all_paths_between_util(int start,
                                   int end,
                                   std::vector<int>& path,
                                   int cutoff);

  /*
   ==========================================================================
   Utilities
   ==========================================================================
  */
  void set_default_initial_state();

  std::mt19937 rand_num_generator;

  // Uniform distribution used by the MCMC sampler
  std::uniform_real_distribution<double> uni_dist;

  // Normal distrubution used to perturb β
  std::normal_distribution<double> norm_dist;

  int get_vertex_id_for_concept(std::string concept, std::string caller);

  int get_degree(int vertex_id);

  void remove_node(int node_id);

  public:
  ~AnalysisGraph() {}

  /**
   * A method to construct an AnalysisGraph object given a JSON-serialized list
   * of INDRA statements.
   *
   * @param filename: The path to the file containing the JSON-serialized INDRA
   * statements.
   */
  static AnalysisGraph from_json_file(std::string filename,
                                      double belief_score_cutoff = 0.9,
                                      double grounding_score_cutoff = 0.0);

  /**
   * A method to construct an AnalysisGraph object given from a std::vector of
   * ( subject, object ) pairs (Statements)
   *
   * @param statements: A std::vector of CausalFragment objects
   */
  static AnalysisGraph
  from_causal_fragments(std::vector<CausalFragment> causal_fragments);

  // TODO Change the name of this function to something better, like
  // restrict_to_subgraph_for_concept, update docstring

  /**
   * Returns a new AnaysisGraph related to the concept provided,
   * which is a subgraph of this graph.
   *
   * @param concept: The concept where the subgraph is about.
   * @param depth  : The maximum number of hops from the concept provided
   *                 to be included in the subgraph.
   * #param inward : Sets the direction of the causal influence flow to
   *                 examine.
   *                 False - (default) A subgraph rooted at the concept
   * provided.
   *                 True  - A subgraph with all the paths ending at the concept
   * provided.
   */
  AnalysisGraph get_subgraph_for_concept(std::string concept,
                                         bool inward = false,
                                         int depth = -1);

  /**
   * Returns a new AnaysisGraph related to the source concept and the target
   * concetp provided, which is a subgraph of this graph.
   * This subgraph contains all the simple directed paths of length less than
   * or equal to the provided cutoff.
   *
   * @param source_concept: The concept where the influence starts.
   * @param target_concept: The concept where the influence ends.
   * @param cutoff        : Maximum length of a directed simple path from
   *                        the source to target to be included in the
   *                        subgraph.
   */
  AnalysisGraph get_subgraph_for_concept_pair(std::string source_concept,
                                              std::string target_concept,
                                              int cutoff = -1);

  void prune(int cutoff = 2);

  void add_node(std::string concept);

  void add_edge(CausalFragment causal_fragment);
  void change_polarity_of_edge(std::string source_concept,
                               int source_polarity,
                               std::string target_concept,
                               int target_polarity);
  void remove_node(std::string concept);

  // Note:
  //      Although just calling this->remove_node(concept) within the loop
  //          for( std::string concept : concept_s )
  //      is suffifient to implement this method, it is not very efficient.
  //      It re-calculates directed simple paths for each vertex removed
  //
  //      Therefore, the code in this->remove_node() has been duplicated with
  //      slightly different flow to achive a more efficient execution.
  void remove_nodes(std::unordered_set<std::string> concepts);

  void remove_edge(std::string src, std::string tgt);

  void remove_edges(std::vector<std::pair<std::string, std::string>> edges);

  auto edges() { return boost::make_iterator_range(boost::edges(graph)); }

  /** Number of nodes in the graph */
  int num_nodes() { return boost::num_vertices(graph); }

  auto predecessors(int i) {
    return boost::make_iterator_range(boost::inv_adjacent_vertices(i, graph));
  }

  // Merge node n1 into node n2, with the option to specify relative polarity.
  // void
  // merge_nodes_old(std::string n1, std::string n2, bool same_polarity = true);

  /**
   * Merges the CAG nodes for the two concepts concept_1 and concept_2
   * with the option to specify relative polarity.
   */
  void merge_nodes(std::string concept_1,
                   std::string concept_2,
                   bool same_polarity = true);

  auto out_edges(int i) {
    return boost::make_iterator_range(boost::out_edges(i, graph));
  }

  double get_beta(std::string source_vertex_name,
                  std::string target_vertex_name) {
    // This is ∂target / ∂source
    // return this->A_original(2 * this->name_to_vertex[target_vertex_name],
    //                        2 * this->name_to_vertex[source_vertex_name] + 1);
    return this->A_original(
        2 * get_vertex_id_for_concept(target_vertex_name, "get_beta()"),
        2 * get_vertex_id_for_concept(source_vertex_name, "get_beta()") + 1);
  }

  void construct_beta_pdfs();

  AnalysisGraph
  find_all_paths_for_concept(std::string concept, int depth, bool reverse);

  /*
   * Find all the simple paths between all the paris of nodes of the graph
   */
  void find_all_paths();

  /*
   * Prints the simple paths found between all pairs of nodes of the graph
   * Groupd according to the starting and ending vertex.
   * find_all_paths() should be called before this to populate the paths
   */
  void print_all_paths();

  // Given an edge (source, target vertex ids - i.e. a β ≡ ∂target/∂source),
  // print all the transition matrix cells that are dependent on it.
  void print_cells_affected_by_beta(int source, int target);

  /*
   ==========================================================================
   Sampling and inference
   ----------------------

   This section contains code for sampling and Bayesian inference.
   ==========================================================================
  */

  // Sample elements of the stochastic transition matrix from the
  // prior distribution, based on gradable adjectives.
  void sample_initial_transition_matrix_from_prior();

  /**
   * Utility function that converts a time range given a start date and end date
   * into an integer value.
   * At the moment returns the number of months withing the time range.
   * This should be the number of traing data time points we have
   *
   * @param start_year  : Start year of the training data sequence
   * @param start_month : Start month of the training data sequence
   * @param end_year    : End year of the training data sequence
   * @param end_month   : End month of the training data sequence
   *
   * @return            : Number of months in the training data sequence
   *                      Including both start and end months
   */
  int calculate_num_timesteps(int start_year,
                              int start_month,
                              int end_year,
                              int end_month);

  /**
   * Get the observed state (values for all the indicators)
   * for a given time point from data.
   * See data.hpp::get_data_value() for missing data rules.
   * Note: units are automatically set according
   * to the parameterization of the given CAG.
   *
   * @param year    : Year of the time point data is extracted
   * @param month   : Month of the time point data is extracted
   * @param country : Country where the data is about
   * @param state   : State where the data is about
   *
   * @return        : Observed state std::vector for the specified location
   *                  on the specified time point.
   *                  Access it as: [ vertex id ][ indicator id ]
   */
  std::vector<std::vector<double>>
  get_observed_state_from_data(int year,
                               int month,
                               std::string country = "South Sudan",
                               std::string state = "");

  /**
   * Set the observed state sequence for a given time range from data.
   * The sequence includes both ends of the range.
   * See data.hpp::get_data_value() for missing data rules.
   * Note: units are automatically set according
   * to the parameterization of the given CAG.
   *
   * @param start_year  : Start year of the sequence of data
   * @param start_month : Start month of the sequence of data
   * @param end_year    : End year of the sequence of data
   * @param end_month   : End month of the sequenec of data
   * @param country     : Country where the data is about
   * @param state       : State where the data is about
   *
   */
  void
  set_observed_state_sequence_from_data(int start_year,
                                        int start_month,
                                        int end_year,
                                        int end_month,
                                        std::string country = "South Sudan",
                                        std::string state = "");

  /**
   * Utility function that sets an initial latent state from observed data.
   * This is used for the inference of the transition matrix as well as the
   * training latent state sequences.
   *
   * @param timestep: Optional setting for setting the initial state to be other
   *                  than the first time step. Not currently used.
   *                  0 <= timestep < this->n_timesteps
   */
  void set_initial_latent_state_from_observed_state_sequence(int timestep = 0);

  void initialize_random_number_generator();

  void set_random_initial_latent_state();

  /**
   * To help experiment with initializing βs to differet values
   *
   * @param ib: Criteria to initialize β
   */
  void init_betas_to(InitialBeta ib = InitialBeta::MEAN);

  /**
   * Train a prediction model given a CAG with indicators
   *
   * @param start_year  : Start year of the sequence of data
   * @param start_month : Start month of the sequence of data
   * @param end_year    : End year of the sequence of data
   * @param end_month   : End month of the sequenec of data
   * @param res         : Sampling resolution. The number of samples to retain.
   * @param burn        : Number of samples to throw away. Start retaining
   *                      samples after throwing away this many samples.
   * @param country     : Country where the data is about
   * @param state       : State where the data is about
   * @param units       : Units for each indicator. Maps
   *                      indicator name --> unit
   * @param initial_beta: Criteria to initialize β
   *
   */
  void train_model(int start_year = 2012,
                   int start_month = 1,
                   int end_year = 2017,
                   int end_month = 12,
                   int res = 200,
                   int burn = 10000,
                   std::string country = "South Sudan",
                   std::string state = "",
                   std::map<std::string, std::string> units = {},
                   InitialBeta initial_beta = InitialBeta::ZERO);

  /**
   * Utility function that sets initial latent states for predictions.
   * During model training a latent state sequence is inferred for each sampled
   * transition matrix, these are then used as the initial states for
   * predictions.
   *
   * @param timestep: Optional setting for setting the initial state to be other
                      than the first time step. Ensures that the correct
                      initial state is used.
                      0 <= timestep < this->n_timesteps
   */
  void set_initial_latent_states_for_prediction(int timestep);

  /**
   * Sample a collection of observed state sequences from the likelihood
   * model given a collection of transition matrices.
   *
   * @param timesteps: The number of timesteps for the sequences.
   */
  void sample_predicted_latent_state_sequence_s_from_likelihood(int timesteps);

  /** Generate predicted observed state sequenes given predicted latent state
   * sequences using the emission model
   */
  void
  generate_predicted_observed_state_sequence_s_from_predicted_latent_state_sequence_s();

  /**
   * Given a trained model, generate this->res number of
   * predicted observed state sequences.
   *
   * @param start_year  : Start year of the prediction
   *                      Should be >= the start year of training
   * @param start_month : Start month of the prediction
   *                      If training and prediction start years are equal
   *                      should be >= the start month of training
   * @param end_year    : End year of the prediction
   * @param end_month   : End month of the prediction
   *
   * @return Predicted observed state (indicator value) sequence for the
   *         prediction period including start and end time points.
   *         This is a tuple.
   *         The first element is a std::vector of std::strings with lables for
   * each time point predicted (year-month). The second element contains
   * predicted values. Access it as: [ sample number ][ time point ][ vertex
   * name ][ indicator name ]
   */
  std::tuple<std::pair<std::pair<int,int>,std::pair<int,int>>,
            std::vector<std::string>,
            std::vector<std::vector<
                std::unordered_map<std::string,
                                   std::unordered_map<std::string, double>>>>>
  generate_prediction(int start_year,
                      int start_month,
                      int end_year,
                      int end_month);

  /**
   * Format the prediction result into a format python callers favor.
   *
   * @param pred_timestes: Number of timesteps in the predicted sequence.
   *
   * @return Re-formatted prediction result.
   *         Access it as:
   *         [ sample number ][ time point ][ vertex name ][ indicator name ]
   */
  std::vector<std::vector<
      std::unordered_map<std::string, std::unordered_map<std::string, double>>>>
  format_prediction_result();

  /**
   * this->generate_prediction() must be called before callign this method.
   * Outputs raw predictions for a given indicator that were generated by
   * generate_prediction(). Each column is a time step and the rows are the
   * samples for that time step.
   *
   * @param indicator: A std::string representing the indicator variable for
   which we
   *                   want predictions for.

   * @return A this->res x this->pred_timesteps dimension 2D array
   *         (std::vector of std::vectors)
   *
  */
  std::vector<std::vector<double>> prediction_to_array(std::string indicator);

  std::vector<Eigen::VectorXd> synthetic_latent_state_sequence;
  // ObservedStateSequence synthetic_observed_state_sequence;
  bool synthetic_data_experiment = false;

  void generate_synthetic_latent_state_sequence_from_likelihood();

  void
  generate_synthetic_observed_state_sequence_from_synthetic_latent_state_sequence();

  std::pair<ObservedStateSequence,
            std::tuple<std::pair<std::pair<int,int>,std::pair<int,int>>,std::vector<std::string>,
                      std::vector<std::vector<std::unordered_map<
                          std::string,
                          std::unordered_map<std::string, double>>>>>>
  test_inference_with_synthetic_data(
      int start_year = 2015,
      int start_month = 1,
      int end_year = 2015,
      int end_month = 12,
      int res = 100,
      int burn = 900,
      std::string country = "South Sudan",
      std::string state = "",
      std::map<std::string, std::string> units = {},
      InitialBeta initial_beta = InitialBeta::HALF);

  // TODO: Need testing
  /**
   * Sample observed state std::vector.
   * This is the implementation of the emission function.
   *
   * @param latent_state: Latent state std::vector.
   *                      This has 2 * number of vertices in the CAG.
   *                      Even indices track the state of each vertex.
   *                      Odd indices track the state of the derivative.
   *
   * @return Observed state std::vector. Observed state for each indicator for
   * each vertex. Indexed by: [ vertex id ][ indicator id ]
   */
  std::vector<std::vector<double>>
  sample_observed_state(Eigen::VectorXd latent_state);

  /**
   * Find all the transition matrix (A) cells that are dependent on the β
   * attached to the provided edge and update them.
   * Acts upon this->A_original
   *
   * @param e: The directed edge ≡ β that has been perturbed
   */
  void update_transition_matrix_cells(
      boost::graph_traits<DiGraph>::edge_descriptor e);

  /**
   * Sample a new transition matrix from the proposal distribution,
   * given a current candidate transition matrix.
   * In practice, this amounts to:
   *    Selecting a random β.
   *    Perturbing it a bit.
   *    Updating all the transition matrix cells that are dependent on it.
   */
  // TODO: Need testng
  // TODO: Before calling sample_from_proposal() we must call
  // AnalysisGraph::find_all_paths()
  // TODO: Before calling sample_from_proposal(), we mush assign initial βs and
  // run Tran_Mat_Cell::compute_cell() to initialize the first transistion
  // matrix.
  // TODO: Update Tran_Mat_Cell::compute_cell() to calculate the proper value.
  // At the moment it just computes sum of length of all the paths realted to
  // this cell
  void sample_from_proposal();

  void set_latent_state_sequence();

  double log_normpdf(double x, double mean, double sd);

  void set_log_likelihood();

  double calculate_delta_log_prior();

  void revert_back_to_previous_state();

  /**
   * Run Bayesian inference - sample from the posterior distribution.
   */
  void sample_from_posterior();

  // ==========================================================================
  // Manipulation
  // ==========================================================================

  void
  set_indicator(std::string concept, std::string indicator, std::string source);

  void delete_indicator(std::string concept, std::string indicator);

  void delete_all_indicators(std::string concept);

  /*
  // TODO: Demosntrate how to use the Node::get_indicator() method
  // with the custom exception.
  // Not sure whether we need this method in AnalaysisGraph
  // so that python side can directly access the Indicator class
  // objects and maipulate them (we need to fiture out how to map a
  // custom class from C++ into python for this) - harder
  // or
  // mirror getter and setter methods of the Indicator class
  // in AnalysisGraph and make the python side call them - easier.
  Indicator get_indicator(std::string concept, std::string indicator) {
    try {
      return graph[name_to_vertex.at(concept)].get_indicator(indicator);
    } catch (const std::out_of_range &oor) {
      fmt::print("Error: AnalysisGraph::get_indicator()\n");
      fmt::print("\tConcept: {} is not in the CAG\n", concept);
    } catch (IndicatorNotFoundException &infe) {
      std::cerr << "Error: AnalysisGraph::get_indicator()\n"
                << "\tindicator: " << infe.what()
                << " is not attached to CAG node " << concept << std::endl;
    }
  }
  */

  void replace_indicator(std::string concept,
                         std::string indicator_old,
                         std::string indicator_new,
                         std::string source);

  /*
    ==========================================================================
    Model parameterization
    *Loren: I am going to try to port this, I'll try not to touch anything up
    top
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
  void map_concepts_to_indicators(int n = 1);

  /**
   * Parameterize the indicators of the AnalysisGraph..
   *
   */
  void parameterize(std::string country = "South Sudan",
                    std::string state = "",
                    int year = 2012,
                    int month = 1,
                    std::map<std::string, std::string> units = {});

  void print_nodes();

  void print_edges();

  void print_name_to_vertex();

  std::pair<Agraph_t*, GVC_t*> to_agraph();

  std::string to_dot();

  void to_png(std::string filename = "CAG.png");

  void print_indicators();
};
