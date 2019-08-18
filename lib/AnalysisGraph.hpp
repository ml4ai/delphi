#pragma once

#include <Eigen/Dense>

#include <boost/graph/graph_traits.hpp>

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
  // Manujinda: I had to move this up since I am usign this within the private:
  // block This is ugly. We need to re-factor the code to make it pretty again
  auto vertices();

  auto successors(int i);

  auto successors(std::string node_name);

  // Allocate a num_verts x num_verts 2D array (std::vector of std::vectors)
  void allocate_A_beta_factors();

  void print_A_beta_factors();

  private:
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

  /**
   * Finds all the simple paths starting at the start vertex and
   * ending at the end vertex.
   * Uses find_all_paths_between_util() as a helper to recursively find the
   * paths
   */
  std::unordered_set<int>
  find_all_paths_between(int start, int end, int cutoff);

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
                                   std::unordered_set<int>& vertices_to_keep,
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
   *                 False - (default) A subgraph rooted at the concept provided. 
   *                 True  - A subgraph with all the paths ending at the concept provided.
   */
  void get_subgraph_for_concept_old(std::string concept,
                                int depth = 1,
                                bool inward = false);

  AnalysisGraph get_subgraph_for_concept_old(std::string concept,
                                             int depth = 1,
                                             bool inward = false);

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
                                              int cutoff);

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

  // auto add_edge(std::string source, std::string target) {
  // boost::add_edge(
  // this->name_to_vertex[source], this->name_to_vertex[target], graph);
  //}

  void remove_edge(std::string src, std::string tgt);

  void remove_edges(std::vector<std::pair<std::string, std::string>> edges);

  auto edges() { return boost::make_iterator_range(boost::edges(graph)); }

  /** Number of nodes in the graph */
  int num_nodes() { return boost::num_vertices(graph); }

  auto predecessors(int i) {
    return boost::make_iterator_range(boost::inv_adjacent_vertices(i, graph));
  }

  auto predecessors(std::string node_name) {
    return predecessors(this->name_to_vertex[node_name]);
  }

  // Merge node n1 into node n2, with the option to specify relative polarity.
  void
  merge_nodes_old(std::string n1, std::string n2, bool same_polarity = true);

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
  void print_all_paths() {
    int num_verts = boost::num_vertices(graph);

    std::cout << "All the simple paths of:" << std::endl;

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
  void print_cells_affected_by_beta(int source, int target) {
    typedef std::multimap<std::pair<int, int>, std::pair<int, int>>::iterator
        MMAPIterator;

    std::pair<int, int> beta = std::make_pair(source, target);

    std::pair<MMAPIterator, MMAPIterator> beta_dept_cells =
        this->beta2cell.equal_range(beta);

    std::cout << std::endl
              << "Cells of A afected by beta_(" << source << ", " << target
              << ")" << std::endl;

    for (MMAPIterator it = beta_dept_cells.first; it != beta_dept_cells.second;
         it++) {
      std::cout << "(" << it->second.first * 2 << ", "
                << it->second.second * 2 + 1 << ") ";
    }
    std::cout << std::endl;
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
  void sample_initial_transition_matrix_from_prior() {
    int num_verts = boost::num_vertices(this->graph);

    // A base transition matrix with the entries that does not change across
    // samples.
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
     *  of rows with index % 3 = 1 would be non zero.
     */
    this->A_original = Eigen::MatrixXd::Identity(num_verts * 2, num_verts * 2);

    // Fill the Δts
    for (int vert = 0; vert < 2 * num_verts; vert += 2) {
      this->A_original(vert, vert + 1) = this->delta_t;
    }

    // Update the β factor dependent cells of this matrix
    for (auto & [ row, col ] : this->beta_dependent_cells) {
      this->A_original(row * 2, col * 2 + 1) =
          // this->A_beta_factors[row][col]->sample_from_prior(this->graph);
          this->A_beta_factors[row][col]->compute_cell(this->graph);
    }
  }

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
                                        std::string state = "") {
    this->observed_state_sequence.clear();

    // Access
    // [ timestep ][ veretx ][ indicator ]
    this->observed_state_sequence = ObservedStateSequence(this->n_timesteps);

    int year = start_year;
    int month = start_month;

    for (int ts = 0; ts < this->n_timesteps; ts++) {
      this->observed_state_sequence[ts] =
          get_observed_state_from_data(year, month, country, state);

      if (month == 12) {
        year++;
        month = 1;
      }
      else {
        month++;
      }
    }
  }

  /**
   * Utility function that sets an initial latent state from observed data.
   * This is used for the inference of the transition matrix as well as the
   * training latent state sequences.
   *
   * @param timestep: Optional setting for setting the initial state to be other
   *                  than the first time step. Not currently used.
   *                  0 <= timestep < this->n_timesteps
   */
  void set_initial_latent_state_from_observed_state_sequence(int timestep = 0) {
    int num_verts = boost::num_vertices(this->graph);

    this->set_default_initial_state();

    for (int v = 0; v < num_verts; v++) {
      std::vector<Indicator>& indicators = this->graph[v].indicators;

      for (int i = 0; i < indicators.size(); i++) {
        Indicator& ind = indicators[i];

        double ind_mean = ind.get_mean();

        while (ind_mean == 0) {
          ind_mean = this->norm_dist(this->rand_num_generator);
        }

        double ind_value = this->observed_state_sequence[timestep][v][i];

        // TODO: If ind_mean is very close to zero, this could overflow
        // Even indexes of the latent state std::vector represent variables
        this->s0_original(2 * v) = ind_value / ind_mean;

        if (timestep == this->n_timesteps - 1) {
          double prev_ind_value =
              this->observed_state_sequence[timestep - 1][v][i];
          double prev_state_value = prev_ind_value / ind_mean;
          double diff = this->s0_original(2 * v) - prev_state_value;
          // TODO: Why this is different from else branch?
          this->s0_original(2 * v + 1) =
              this->norm_dist(this->rand_num_generator) + diff;
        }
        else {
          double next_ind_value =
              this->observed_state_sequence[timestep + 1][v][i];
          double next_state_value = next_ind_value / ind_mean;
          double diff = next_state_value - this->s0_original(2 * v);
          // TODO: Why this is different from if branch?
          this->s0_original(2 * v + 1) = diff;
        }
      }
    }
  }

  void initialize_random_number_generator();

  void set_random_initial_latent_state() {
    int num_verts = boost::num_vertices(this->graph);

    this->set_default_initial_state();

    for (int v = 0; v < num_verts; v++) {
      this->s0_original(2 * v + 1) =
          0.1 * this->uni_dist(this->rand_num_generator);
    }
  }

  /**
   * To help experiment with initializing βs to differet values
   *
   * @param ib: Criteria to initialize β
   */
  void init_betas_to(InitialBeta ib = InitialBeta::MEAN) {
    switch (ib) {
    // Initialize the initial β for this edge
    // Note: I am repeating the loop within each case for efficiency.
    // If we embed the switch withn the for loop, there will be less code
    // but we will evaluate the switch for each iteration through the loop
    case InitialBeta::ZERO:
      for (boost::graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
        graph[e].beta = 0;
      }
      break;
    case InitialBeta::ONE:
      for (boost::graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
        graph[e].beta = 1.0;
      }
      break;
    case InitialBeta::HALF:
      for (boost::graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
        graph[e].beta = 0.5;
      }
      break;
    case InitialBeta::MEAN:
      for (boost::graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
        graph[e].beta = graph[e].kde.value().mu;
      }
      break;
    case InitialBeta::RANDOM:
      for (boost::graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
        // this->uni_dist() gives a random number in range [0, 1]
        // Multiplying by 2 scales the range to [0, 2]
        // Sustracting 1 moves the range to [-1, 1]
        graph[e].beta = this->uni_dist(this->rand_num_generator) * 2 - 1;
      }
      break;
    }
  }

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
  void set_initial_latent_states_for_prediction(int timestep = 0) {
    this->prediction_initial_latent_state_s.clear();
    this->prediction_initial_latent_state_s =
        std::vector<Eigen::VectorXd>(this->res);

    transform(
        this->training_latent_state_sequence_s.begin(),
        this->training_latent_state_sequence_s.end(),
        this->prediction_initial_latent_state_s.begin(),
        [&timestep](std::vector<Eigen::VectorXd>& ls) { return ls[timestep]; });
  }

  /**
   * Sample a collection of observed state sequences from the likelihood
   * model given a collection of transition matrices.
   *
   * @param timesteps: The number of timesteps for the sequences.
   */
  void sample_predicted_latent_state_sequence_s_from_likelihood(int timesteps) {
    this->n_timesteps = timesteps;

    int num_verts = boost::num_vertices(this->graph);

    // Allocate memory for prediction_latent_state_sequence_s
    this->predicted_latent_state_sequence_s.clear();
    this->predicted_latent_state_sequence_s =
        std::vector<std::vector<Eigen::VectorXd>>(
            this->res,
            std::vector<Eigen::VectorXd>(this->n_timesteps,
                                         Eigen::VectorXd(num_verts * 2)));

    for (int samp = 0; samp < this->res; samp++) {
      this->predicted_latent_state_sequence_s[samp][0] =
          this->prediction_initial_latent_state_s[samp];

      for (int ts = 1; ts < this->n_timesteps; ts++) {
        this->predicted_latent_state_sequence_s[samp][ts] =
            this->transition_matrix_collection[samp] *
            this->predicted_latent_state_sequence_s[samp][ts - 1];
      }
    }
  }

  /** Generate predicted observed state sequenes given predicted latent state
   * sequences using the emission model
   */
  void
  generate_predicted_observed_state_sequence_s_from_predicted_latent_state_sequence_s() {
    // Allocate memory for observed_state_sequences
    this->predicted_observed_state_sequence_s.clear();
    this->predicted_observed_state_sequence_s =
        std::vector<ObservedStateSequence>(
            this->res,
            ObservedStateSequence(this->n_timesteps,
                                  std::vector<std::vector<double>>()));

    for (int samp = 0; samp < this->res; samp++) {
      std::vector<Eigen::VectorXd>& sample =
          this->predicted_latent_state_sequence_s[samp];

      transform(sample.begin(),
                sample.end(),
                this->predicted_observed_state_sequence_s[samp].begin(),
                [this](Eigen::VectorXd latent_state) {
                  return this->sample_observed_state(latent_state);
                });
    }
  }

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
  std::pair<std::vector<std::string>,
            std::vector<std::vector<
                std::unordered_map<std::string,
                                   std::unordered_map<std::string, double>>>>>
  generate_prediction(int start_year,
                      int start_month,
                      int end_year,
                      int end_month) {
    if (!this->trained) {
      fmt::print("Passed untrained Causal Analysis Graph (CAG) Model. \n",
                 "Try calling <CAG>.train_model(...) first!");
      throw "Model not yet trained";
    }

    if (start_year < this->init_training_year ||
        (start_year == this->init_training_year &&
         start_month < this->init_training_month)) {
      fmt::print("The initial prediction date can't be before the\n"
                 "inital training date. Defaulting initial prediction date\n"
                 "to initial training date.");
      start_year = this->init_training_year;
      start_month = this->init_training_month;
    }

    /*
     *              total_timesteps
     *   ____________________________________________
     *  |                                            |
     *  v                                            v
     * start trainig                                 end prediction
     *  |--------------------------------------------|
     *  :           |--------------------------------|
     *  :         start prediction                   :
     *  ^           ^                                ^
     *  |___________|________________________________|
     *      diff              pred_timesteps
     */
    int total_timesteps =
        this->calculate_num_timesteps(this->init_training_year,
                                      this->init_training_month,
                                      end_year,
                                      end_month);

    this->pred_timesteps = this->calculate_num_timesteps(
        start_year, start_month, end_year, end_month);

    int diff_timesteps = total_timesteps - this->pred_timesteps;
    int truncate = 0;

    int year = start_year;
    int month = start_month;

    this->pred_range.clear();
    this->pred_range = std::vector<std::string>(this->pred_timesteps);

    for (int ts = 0; ts < this->pred_timesteps; ts++) {
      this->pred_range[ts] = std::to_string(year) + "-" + std::to_string(month);

      if (month == 12) {
        year++;
        month = 1;
      }
      else {
        month++;
      }
    }

    int pred_init_timestep = diff_timesteps;

    if (diff_timesteps >= this->n_timesteps) {
      /*
       *              total_timesteps
       *   ____________________________________________
       *  |                                            |
       *  v                                            v
       *tr_st           tr_ed     pr_st              pr_ed
       *  |---------------|---------|------------------|
       *  ^               ^         ^                  ^
       *  |_______________|_________|__________________|
       *  :  n_timesteps   truncate :   pred_timesteps
       *  ^                         ^
       *  |_________________________|
       *            diff
       */
      pred_init_timestep = this->n_timesteps - 1;
      truncate = diff_timesteps - this->n_timesteps;
      this->pred_timesteps += truncate; // = total_timesteps - this->n_timesteps
    }

    this->set_initial_latent_states_for_prediction(pred_init_timestep);

    this->sample_predicted_latent_state_sequence_s_from_likelihood(
        this->pred_timesteps);
    this->generate_predicted_observed_state_sequence_s_from_predicted_latent_state_sequence_s();

    for (std::vector<Eigen::VectorXd>& latent_state_s :
         this->predicted_latent_state_sequence_s) {
      latent_state_s.erase(latent_state_s.begin(),
                           latent_state_s.begin() + truncate);
    }

    for (ObservedStateSequence& observed_state_s :
         this->predicted_observed_state_sequence_s) {
      observed_state_s.erase(observed_state_s.begin(),
                             observed_state_s.begin() + truncate);
    }

    this->pred_timesteps -= truncate;
    return std::make_pair(this->pred_range, this->format_prediction_result());
  }

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
  format_prediction_result() {
    // Access
    // [ sample ][ time_step ][ vertex_name ][ indicator_name ]
    std::vector<std::vector<
        std::unordered_map<std::string,
                           std::unordered_map<std::string, double>>>>
        result = std::vector<std::vector<
            std::unordered_map<std::string,
                               std::unordered_map<std::string, double>>>>(
            this->res,
            std::vector<
                std::unordered_map<std::string,
                                   std::unordered_map<std::string, double>>>(
                this->pred_timesteps));

    for (int samp = 0; samp < this->res; samp++) {
      for (int ts = 0; ts < this->pred_timesteps; ts++) {
        for (auto[vert_name, vert_id] : this->name_to_vertex) {
          for (auto[ind_name, ind_id] : this->graph[vert_id].indicator_names) {
            result[samp][ts][vert_name][ind_name] =
                this->predicted_observed_state_sequence_s[samp][ts][vert_id]
                                                         [ind_id];
          }
        }
      }
    }

    return result;
  }

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
  std::vector<std::vector<double>> prediction_to_array(std::string indicator) {
    int vert_id = -1;
    int ind_id = -1;

    std::vector<std::vector<double>> result = std::vector<std::vector<double>>(
        this->res, std::vector<double>(this->pred_timesteps));

    // Find the vertex id the indicator is attached to and
    // the indicator id of it.
    // TODO: We can make this more efficient by making indicators_in_CAG
    // a map from indicator names to vertices they are attached to.
    // This is just a quick and dirty implementation
    for (auto[v_name, v_id] : this->name_to_vertex) {
      for (auto[i_name, i_id] : this->graph[v_id].indicator_names) {
        if (indicator.compare(i_name) == 0) {
          vert_id = v_id;
          ind_id = i_id;
          goto indicator_found;
        }
      }
    }
    // Program will reach hear only if the indicator is not found
    fmt::print("AnalysisGraph::prediction_to_array - indicator {} not found!\n",
               indicator);
    throw IndicatorNotFoundException(fmt::format(
        "AnalysisGraph::prediction_to_array - indicator \"{}\" not found!\n",
        indicator));

  indicator_found:

    for (int samp = 0; samp < this->res; samp++) {
      for (int ts = 0; ts < this->pred_timesteps; ts++) {
        result[samp][ts] =
            this->predicted_observed_state_sequence_s[samp][ts][vert_id]
                                                     [ind_id];
      }
    }

    return result;
  }

  std::vector<Eigen::VectorXd> synthetic_latent_state_sequence;
  // ObservedStateSequence synthetic_observed_state_sequence;
  bool synthetic_data_experiment = false;

  void generate_synthetic_latent_state_sequence_from_likelihood() {
    int num_verts = boost::num_vertices(this->graph);

    // Allocate memory for synthetic_latent_state_sequence
    this->synthetic_latent_state_sequence.clear();
    this->synthetic_latent_state_sequence = std::vector<Eigen::VectorXd>(
        this->n_timesteps, Eigen::VectorXd(num_verts * 2));

    this->synthetic_latent_state_sequence[0] = this->s0_original;

    for (int ts = 1; ts < this->n_timesteps; ts++) {
      this->synthetic_latent_state_sequence[ts] =
          this->A_original * this->synthetic_latent_state_sequence[ts - 1];
    }
  }

  void
  generate_synthetic_observed_state_sequence_from_synthetic_latent_state_sequence() {
    // Allocate memory for observed_state_sequences
    this->observed_state_sequence.clear();
    this->observed_state_sequence = ObservedStateSequence(
        this->n_timesteps, std::vector<std::vector<double>>());

    transform(this->synthetic_latent_state_sequence.begin(),
              this->synthetic_latent_state_sequence.end(),
              this->observed_state_sequence.begin(),
              [this](Eigen::VectorXd latent_state) {
                return this->sample_observed_state(latent_state);
              });
  }

  std::pair<ObservedStateSequence,
            std::pair<std::vector<std::string>,
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
      InitialBeta initial_beta = InitialBeta::HALF) {

    synthetic_data_experiment = true;

    this->n_timesteps = this->calculate_num_timesteps(
        start_year, start_month, end_year, end_month);
    this->init_betas_to(initial_beta);
    this->sample_initial_transition_matrix_from_prior();
    std::cout << this->A_original << std::endl;
    this->parameterize(country, state, start_year, start_month, units);

    // Initialize the latent state std::vector at time 0
    this->set_random_initial_latent_state();
    this->generate_synthetic_latent_state_sequence_from_likelihood();
    this->generate_synthetic_observed_state_sequence_from_synthetic_latent_state_sequence();

    for (std::vector<std::vector<double>> obs : this->observed_state_sequence) {
      fmt::print("({}, {})\n", obs[0][0], obs[1][0]);
    }

    this->train_model(start_year,
                      start_month,
                      end_year,
                      end_month,
                      res,
                      burn,
                      country,
                      state,
                      units,
                      InitialBeta::ZERO);

    return std::make_pair(this->observed_state_sequence,
                          this->generate_prediction(
                              start_year, start_month, end_year, end_month));

    synthetic_data_experiment = false;
  }

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
  sample_observed_state(Eigen::VectorXd latent_state) {
    int num_verts = boost::num_vertices(this->graph);

    assert(num_verts == latent_state.size() / 2);

    std::vector<std::vector<double>> observed_state(num_verts);

    for (int v = 0; v < num_verts; v++) {
      std::vector<Indicator>& indicators = this->graph[v].indicators;

      observed_state[v] = std::vector<double>(indicators.size());

      // Sample observed value of each indicator around the mean of the
      // indicator
      // scaled by the value of the latent state that caused this observation.
      // TODO: Question - Is ind.mean * latent_state[ 2*v ] correct?
      //                  Shouldn't it be ind.mean + latent_state[ 2*v ]?
      transform(indicators.begin(),
                indicators.end(),
                observed_state[v].begin(),
                [&](Indicator ind) {
                  std::normal_distribution<double> gaussian(
                      ind.mean * latent_state[2 * v], ind.stdev);

                  return gaussian(this->rand_num_generator);
                });
    }

    return observed_state;
  }

  /**
   * Find all the transition matrix (A) cells that are dependent on the β
   * attached to the provided edge and update them.
   * Acts upon this->A_original
   *
   * @param e: The directed edge ≡ β that has been perturbed
   */
  void update_transition_matrix_cells(
      boost::graph_traits<DiGraph>::edge_descriptor e) {
    std::pair<int, int> beta = std::make_pair(boost::source(e, this->graph),
                                              boost::target(e, this->graph));

    typedef std::multimap<std::pair<int, int>, std::pair<int, int>>::iterator
        MMAPIterator;

    std::pair<MMAPIterator, MMAPIterator> beta_dept_cells =
        this->beta2cell.equal_range(beta);

    // TODO: I am introducing this to implement calculate_Δ_log_prior
    // Remember the cells of A that got changed and their previous values
    // this->A_cells_changed.clear();

    for (MMAPIterator it = beta_dept_cells.first; it != beta_dept_cells.second;
         it++) {
      int& row = it->second.first;
      int& col = it->second.second;

      // Note that I am remembering row and col instead of 2*row and 2*col+1
      // row and col resembles an edge in the CAG: row -> col
      // ( 2*row, 2*col+1 ) is the transition mateix cell that got changed.
      // this->A_cells_changed.push_back( make_tuple( row, col, A( row * 2, col
      // * 2 + 1 )));

      this->A_original(row * 2, col * 2 + 1) =
          this->A_beta_factors[row][col]->compute_cell(this->graph);
    }
  }

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
  void sample_from_proposal() {
    // Randomly pick an edge ≡ β
    boost::iterator_range edge_it = this->edges();

    std::vector<boost::graph_traits<DiGraph>::edge_descriptor> e(1);
    sample(
        edge_it.begin(), edge_it.end(), e.begin(), 1, this->rand_num_generator);

    // Remember the previous β
    this->previous_beta = std::make_pair(e[0], this->graph[e[0]].beta);

    // Perturb the β
    // TODO: Check whether this perturbation is accurate
    graph[e[0]].beta += this->norm_dist(this->rand_num_generator);

    this->update_transition_matrix_cells(e[0]);
  }

  void set_latent_state_sequence() {
    int num_verts = boost::num_vertices(this->graph);

    // Allocate memory for latent_state_sequence
    this->latent_state_sequence.clear();
    this->latent_state_sequence =
        std::vector<Eigen::VectorXd>(this->n_timesteps);

    this->latent_state_sequence[0] = this->s0_original;

    for (int ts = 1; ts < this->n_timesteps; ts++) {
      this->latent_state_sequence[ts] =
          this->A_original * this->latent_state_sequence[ts - 1];
    }
  }

  double log_normpdf(double x, double mean, double sd) {
    double var = pow(sd, 2);
    double log_denom = -0.5 * log(2 * M_PI) - log(sd);
    double log_nume = pow(x - mean, 2) / (2 * var);

    return log_denom - log_nume;
  }

  void set_log_likelihood();

  double calculate_delta_log_prior() {
    // If kde of an edge is truely optional ≡ there are some
    // edges without a kde assigned, we should not access it
    // using .value() (In the case of kde being missing, this
    // will throw an exception). We should follow a process
    // similar to Tran_Mat_Cell::sample_from_prior
    KDE& kde = this->graph[this->previous_beta.first].kde.value();

    // We have to return: log( p( β_new )) - log( p( β_old ))
    return kde.logpdf(this->graph[this->previous_beta.first].beta) -
           kde.logpdf(this->previous_beta.second);
  }

  void revert_back_to_previous_state() {
    this->log_likelihood = this->previous_log_likelihood;

    this->graph[this->previous_beta.first].beta = this->previous_beta.second;

    // Reset the transition matrix cells that were changed
    // TODO: Can we change the transition matrix only when the sample is
    // accpeted?
    this->update_transition_matrix_cells(this->previous_beta.first);
  }

  /**
   * Run Bayesian inference - sample from the posterior distribution.
   */
  void sample_from_posterior() {
    // Sample a new transition matrix from the proposal distribution
    this->sample_from_proposal();

    double delta_log_prior = this->calculate_delta_log_prior();

    this->set_log_likelihood();
    double delta_log_likelihood =
        this->log_likelihood - this->previous_log_likelihood;

    double delta_log_joint_probability = delta_log_prior + delta_log_likelihood;

    double acceptance_probability =
        std::min(1.0, exp(delta_log_joint_probability));

    if (acceptance_probability < uni_dist(this->rand_num_generator)) {
      // Reject the sample
      this->revert_back_to_previous_state();
    }
  }

  // ==========================================================================
  // Manipulation
  // ==========================================================================

  void set_indicator(std::string concept,
                     std::string indicator,
                     std::string source) {
    if (this->indicators_in_CAG.find(indicator) !=
        this->indicators_in_CAG.end()) {
      fmt::print(
          "{0} already exists in Casual Analysis Graph, Indicator {0} was "
          "not added to Concept {1}.",
          indicator,
          concept);
      return;
    }
    try {
      this->graph[this->name_to_vertex.at(concept)].add_indicator(indicator,
                                                                  source);
      this->indicators_in_CAG.insert(indicator);
    }
    catch (const std::out_of_range& oor) {
      std::cerr << "Error: AnalysisGraph::set_indicator()\n"
                << "\tConcept: " << concept << " is not in the CAG\n";
      std::cerr << "\tIndicator: " << indicator << " with Source: " << source
                << std::endl;
      std::cerr << "\tCannot be added\n";
    }
  }

  void delete_indicator(std::string concept, std::string indicator) {
    try {
      this->graph[this->name_to_vertex.at(concept)].delete_indicator(indicator);
      this->indicators_in_CAG.erase(indicator);
    }
    catch (const std::out_of_range& oor) {
      std::cerr << "Error: AnalysisGraph::delete_indicator()\n"
                << "\tConcept: " << concept << " is not in the CAG\n";
      std::cerr << "\tIndicator: " << indicator << " cannot be deleted"
                << std::endl;
    }
  }

  void delete_all_indicators(std::string concept) {
    try {
      this->graph[this->name_to_vertex.at(concept)].clear_indicators();
    }
    catch (const std::out_of_range& oor) {
      std::cerr << "Error: AnalysisGraph::delete_indicator()\n"
                << "\tConcept: " << concept << " is not in the CAG\n";
      std::cerr << "\tIndicators cannot be deleted" << std::endl;
    }
  }
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
                         std::string source) {

    if (this->indicators_in_CAG.find(indicator_new) !=
        this->indicators_in_CAG.end()) {
      fmt::print(
          "{0} already exists in Casual Analysis Graph, Indicator {0} did "
          "not replace Indicator {1} for Concept {2}.",
          indicator_new,
          indicator_old,
          concept);
      return;
    }

    try {
      this->graph[this->name_to_vertex.at(concept)].replace_indicator(
          indicator_old, indicator_new, source);
      this->indicators_in_CAG.insert(indicator_new);
      this->indicators_in_CAG.erase(indicator_old);
    }
    catch (const std::out_of_range& oor) {
      std::cerr << "Error: AnalysisGraph::replace_indicator()\n"
                << "\tConcept: " << concept << " is not in the CAG\n";
      std::cerr << "\tIndicator: " << indicator_old << " cannot be replaced"
                << std::endl;
    }
  }

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
