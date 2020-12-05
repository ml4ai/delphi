#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <boost/graph/graph_traits.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/iterator_range.hpp>

#include "graphviz_interface.hpp"

#include "DiGraph.hpp"
#include "Tran_Mat_Cell.hpp"
#include <fmt/format.h>
#include <nlohmann/json.hpp>

const double tuning_param = 1.0;

enum InitialBeta { ZERO, ONE, HALF, MEAN, RANDOM };

typedef std::unordered_map<std::string, std::vector<double>>
    AdjectiveResponseMap;

// This is a multimap to keep provision to have multiple observations per
// time point per indicator.
// Access (concept is a vertex in the CAG)
// [ concept ][ indicator ][ <year, month> --→ observation ]
typedef std::vector<std::vector<std::multimap<std::pair<int, int>, double>>>
    ConceptIndicatorData;

// Keeps the sequence of dates for which data points are available
// Data points are sorted according to dates
// Access:
// [ concept ][ indicator ][<year, month>]
typedef std::vector<std::vector<std::pair<int, int>>> ConceptIndicatorDates;

// Access
// [ timestep ][ concept ][ indicator ][ observation ]
typedef std::vector<std::vector<std::vector<std::vector<double>>>>
    ObservedStateSequence;

typedef std::vector<std::vector<std::vector<double>>>
    PredictedObservedStateSequence;

typedef std::pair<std::tuple<std::string, int, std::string>,
                  std::tuple<std::string, int, std::string>>
    CausalFragment;

// Access
// [ sample ][ time_step ]{ vertex_name --> { indicator_name --> pred}}
// [ sample ][ time_step ][ vertex_name ][ indicator_name ]
typedef std::vector<std::vector<
    std::unordered_map<std::string, std::unordered_map<std::string, double>>>>
    FormattedPredictionResult;

// Access
// [ vertex_name ][ timestep ][ sample ]
typedef std::unordered_map<std::string, std::vector<std::vector<double>>>
    FormattedProjectionResult;

// Access
// get<0>:
//      Training range
//      <<start_year, start_month>, <end_year, end_month>>
// get<1>:
//      Sequence of prediction time steps
//      [yyyy-mm₀, yyyy-mm₁, yyyy-mm₂, yyyy-mm₃, .....]
// get<2>:
//      Prediction results
//      [ sample ][ time_step ]{ vertex_name --> { indicator_name --> pred}}
//      [ sample ][ time_step ][ vertex_name ][ indicator_name ]
typedef std::tuple<std::pair<std::pair<int, int>, std::pair<int, int>>,
                   std::vector<std::string>,
                   FormattedPredictionResult>
    Prediction;

// Access
// [prediction time step] -->
//      get<0>:
//          concept name
//      get<1>
//          indicator name
//      get<2>
//          value
typedef std::unordered_map<int, std::vector<std::tuple<std::string,
        std::string, double>>> ConstraintSchedule;

typedef boost::graph_traits<DiGraph>::edge_descriptor EdgeDescriptor;
typedef boost::graph_traits<DiGraph>::edge_iterator EdgeIterator;

typedef std::multimap<std::pair<int, int>, std::pair<int, int>>::iterator
    MMapIterator;

AdjectiveResponseMap construct_adjective_response_map(size_t n_kernels);

/**
 * The AnalysisGraph class is the main model/interface for Delphi.
 */
class AnalysisGraph {

  private:
  // True only when Delphi is run through the CauseMos HMI.
  bool causemos_call = false;

  DiGraph graph;

  // Handle to the random number generator singleton object
  RNG* rng_instance = nullptr;

  std::mt19937 rand_num_generator;


  // Uniform distribution used by the MCMC sampler
  std::uniform_real_distribution<double> uni_dist;

  // Normal distribution used to perturb β
  std::normal_distribution<double> norm_dist;

  // Uniform discrete distribution used by the MCMC sampler
  // to perturb the initial latent state
  std::uniform_int_distribution<int> uni_disc_dist;

  // Sampling resolution
  size_t res;

  /*
   ============================================================================
   Meta Data Structures
   ============================================================================
  */

  // Maps each concept name to the vertex id of the
  // vertex that concept is represented in the CAG
  // concept name --> CAG vertex id
  std::unordered_map<std::string, int> name_to_vertex = {};

  // Keeps track of indicators in CAG to ensure there are no duplicates.
  std::unordered_set<std::string> indicators_in_CAG;

  // A_beta_factors is a 2D array (std::vector of std::vectors) that keeps track
  // of the β factors involved with each cell of the transition matrix A.
  //
  // According to our current model, which uses variables and their partial
  // derivatives with respect to each other ( x --> y, βxy = ∂y/∂x ),
  // at most half of the transition matrix cells can be affected by βs.
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

  /*
   ============================================================================
   Sampler Related Variables
   ============================================================================
  */

  // Keep track whether the model is trained.
  // Used to check whether there is a trained model before calling
  // generate_prediction()
  bool trained = false;

  int n_timesteps = 0;
  int pred_timesteps = 0;
  std::pair<std::pair<int, int>, std::pair<int, int>> training_range;
  std::vector<std::string> pred_range;

  double t = 0.0;
  double delta_t = 1.0;

  double log_likelihood = 0.0;
  double previous_log_likelihood = 0.0;

  // To decide whether to perturb a θ or a derivative
  // If coin_flip < coin_flip_thresh perturb θ else perturb derivative
  double coin_flip = 0;
  double coin_flip_thresh = 0.5;

  // Remember the old θ and the edge where we perturbed the θ.
  // We need this to revert the system to the previous state if the proposal
  // gets rejected.
  std::pair<EdgeDescriptor, double> previous_theta;

  // Remember the old derivative and the concept we perturbed the derivative
  int changed_derivative = 0;
  double previous_derivative = 0;

  // Latent state that is evolved by sampling.
  Eigen::VectorXd s0;
  Eigen::VectorXd s0_prev;

  // Transition matrix that is evolved by sampling.
  // Since variable A has been already used locally in other methods,
  // I chose to name this A_original. After refactoring the code, we could
  // rename this to A.
  Eigen::MatrixXd A_original;

  // Determines whether to use the continuous version or the discretized
  // version of the solution for the system of differential equations.
  //
  // continuous = true:
  //    Continuous version of the solution. We use the continuous form of the
  //    transition matrix and matrix exponential.
  //
  // continuous = false:
  //    Discretized version of the solution. We use the discretized version of
  //    the transition matrix and repeated matrix multiplication.
  //
  // A_discretized = I + A_continuous * Δt
  bool continuous = true;

  // Access this as
  // current_latent_state
  Eigen::VectorXd current_latent_state;

  // Access this as
  // observed_state_sequence[ time step ][ vertex ][ indicator ]
  ObservedStateSequence observed_state_sequence;

  // Access this as
  // prediction_latent_state_sequences[ sample ][ time step ]
  std::vector<std::vector<Eigen::VectorXd>> predicted_latent_state_sequences;

  // Access this as
  // predicted_observed_state_sequences
  //                            [ sample ][ time step ][ vertex ][ indicator ]
  std::vector<PredictedObservedStateSequence>
      predicted_observed_state_sequences;

  PredictedObservedStateSequence test_observed_state_sequence;

  // Implementing constraints or interventions.
  // -------------------------------------------------------------------------
  // We are implementing two ways to constrain the model.
  //    1. One-off constraints.
  //        A latent state is clamped to a constrained value just for the
  //        specified time step and released to evolve from the subsequent time
  //        step onward until the next constrained time step and so on.
  //        E.g. Getting a one time grant.
  //    2. Perpetual constraints
  //        Once a latent state gets clamped at a value at a particular time
  //        step, it stays clamped at that value in subsequent time steps until
  //        another constrain overwrites the current constrain or the end of
  //        the prediction time is reached.
  //        NOTE: Currently we do not have a way to have a semi-perpetual
  //        constraint: A constraint is applied perpetually for some number of
  //        continuous time steps and then switched off. With a little bit of
  //        work we can implement this. We just need a special constraint value
  //        to signal end of a constraint. One suggestion is to use NaN.
  //        E.g. Maintaining a water level of a reservoir at a certain amount.
  //
  // NOTE: WE either apply One-off or Perpetual constraints to all the
  //       concepts. The current design does not permit applying mixed
  //       constraints such that some concepts are constrained one-off while
  //       some others are constrained perpetual. With a little bit more work,
  //       we could also achieve this. Moving the constraint type into the
  //       constraint information data structure would work for keeping track
  //       of mixed constraint types:
  //        std::unordered_map<int, std::vector<std::tuple<int, double, bool>>>
  //       Then we would have to update the constraint processing logic
  //       accordingly.
  // -------------------------------------------------------------------------
  //
  // NOTE: This implementation of the constraints does not work at all with
  // multiple indicators being attached to a single concept. Constraining the
  // concept effects all the indicators and we cannot constrain targeted for a
  // particular indicator. In the current model we might achieve this by
  // constraining the scaling factor (which we incorrectly call as the
  // indicator mean).
  // Currently we are doing:
  //    constrained latent state = constrained indicator value / scaling factor
  // The constraining that might work with multiple indicators per concept:
  //    constrained scaling factor = constrained indicator value / latent state
  // -------------------------------------------------------------------------
  //
  // Implementing the One-off constraints:
  // -------------------------------------------------------------------------
  // To store constraints (or interventions)
  // For some times steps of the prediction range, latent state values could be
  // constrained to a value external from what the LDS predicts that value
  // should be. When prediction happens, if constrains are present at a time
  // step for some concepts, the predicted latent state values for those
  // concepts are overwritten by the constraints supplied in this data
  // structure.
  // Access
  // [ time step ] --> [(concept id, constrained value), ... ]
  // latent_state_constraints.at(time step)
  std::unordered_map<int, std::vector<std::pair<int, double>>>
      one_off_constraints;
  //
  // Implementing Perpetual constraints:
  // -------------------------------------------------------------------------
  // Access
  // [ concept id ] --> constrained value
  // perpetual_constraints.at(concept id)
  std::unordered_map<int, double> perpetual_constraints;
  //
  // Deciding which type of constraints to enforce
  // one_off_constraints is empty => unconstrained prediction
  // is_one_off_constraints = true => One-off constraints
  // is_one_off_constraints = false => Perpetual constraints
  bool is_one_off_constraints = true;
  //
  // Deciding whether to clamp the latent variable or the derivative
  // true  => clamp at derivative
  // false => clamp at latent variable
  bool clamp_at_derivative = true;
  //
  // When we are clamping derivatives the clamp sticks since derivatives never
  // chance in our current model. So for one-off clamping, we have to reset the
  // derivative back to original after the clamping step. This variable
  // remembers the time step to reset the clamped derivatives.
  int rest_derivative_clamp_ts = -1;

  std::vector<Eigen::MatrixXd> transition_matrix_collection;
  std::vector<Eigen::VectorXd> initial_latent_state_collection;

  std::vector<Eigen::VectorXd> synthetic_latent_state_sequence;
  bool synthetic_data_experiment = false;

  /*
   ============================================================================
   Private: Integration with Uncharted's CauseMos interface
                                                  (in causemos_integration.cpp)
   ============================================================================
  */

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            create-model
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

  /** Extracts concept to indicator mapping and the indicator observation
   * sequences from the create model JSON input received from the CauseMose
   * HMI. The JSON input specifies time as POSIX time stamps in milliseconds.
   * Also the JSON input does not mention anything about the observation
   * frequency, missing data points, or whether observation sequences for
   * multiple indicators are time aligned (e.g. Whether they have the same
   * starting and ending time, whether data points for a single indicator are
   * ordered in chronologically increasing order).
   *
   * This method does not assume any of these unspoken qualities. This method
   * reads in the observations from JSON and populate an internal intermediate
   * and temporary data structure time aligning observations.
   *
   * All the parameters except the first are used to return the results back to
   * the caller. The caller should declare these variables and pass them here so
   * that after the execution of this method, the caller can access the results.
   *
   * @param json_indicators         : conceptIndicators portion of the JSON
   *                                  input received from the HMI.
   * @param concept_indicator_data  : This data structure gets filled with
   *                                  chronologically ordered observation
   *                                  sequences for all the indicators,
   *                                  segmented according to the concepts they
   *                                  attach to. This is a temporary
   *                                  intermediate data structure used to time
   *                                  align observation and accumulate multiple
   *                                  observations for an indicator at a time
   *                                  point. Observed state sequence is filled
   *                                  using data in this data structure.
   * @param concept_indicator_dates : This data structure gets filled with
   *                                  year-month date points where observations
   *                                  are available for each indicator. Each
   *                                  indicator gets a separate sequence of
   *                                  chronologically ordered year-months.
   *                                  These are used to asses the best
   *                                  frequency to align observations across
   *                                  all the indicators.
   * @param start_year              : Start year of the observations. Note
   *                                  that there could be some observation
   *                                  sequences that start after start_year,
   *                                  start_month date.
   * @param start_month             : Start month of the observations. Note
   *                                  that there could be some observation
   *                                  sequences that start after start_year,
   *                                  start_month date.
   * @param end_year                : Ending year of the observations. Note
   *                                  that there could be some observation
   *                                  sequences that end before end_year,
   *                                  end_month date.
   * @param end_month               : Ending month of the observations. Note
   *                                  that there could be some observation
   *                                  sequences that end before end_year,
   *                                  end_month date.
   * @returns void
   *
   */
  void extract_concept_indicator_mapping_and_observations_from_json(
                        const nlohmann::json &json_indicators,
                        ConceptIndicatorData &concept_indicator_data,
                        ConceptIndicatorDates &concept_indicator_dates,
                        int &start_year, int &start_month,
                        int &end_year, int &end_month);

  /** Infer the least common observation frequency for all the
   * observation sequences so that they are time aligned starting from the
   * start_year and start_month.
   * At the moment we do not use the information we gather in this method as
   * the rest of the code by default models at a monthly frequency. The
   * advantage of modeling at the least common observation frequency is less
   * missing data points.
   *
   * TODO: We can and might need to make Delphi adapt to least common
   * observation frequency present in the training data. However, to reach
   * that level, we would have to update some of the older code in other
   * functions. One such method is AnalysisGraph::calculate_num_timesteps(),
   * which assumes a monthly frequency when calculating the number of time
   * steps. We also have to update the plotting functions. There could be other
   * functions I do not foresee that needs updating.
   *
   * NOTE: Some thought about how to use this information:
   * shortest_gap = longest_gap = 1  ⇒ monthly with no missing data
   * shortest_gap = longest_gap = 12 ⇒ yearly with no missing data
   * shortest_gap = longest_gap ≠ 1 or 12  ⇒ no missing data odd frequency
   * shortest_gap = 1 < longest_gap ⇒ monthly with missing data
   *    frequent_gap = 1 ⇒ little missing data
   *    frequent_gap > 1 ⇒ lot of missing data
   * 1 < shortest_gap < longest_gap
   *    Best frequency to model at is the greatest common divisor of all
   *    gaps. For example if we see gaps 4, 6, 10 then gcd(4, 6, 10) = 2
   *    and modeling at a frequency of 2 months starting from the start
   *    date would allow us to capture all the observation sequences while
   *    aligning them with each other.
   *
   * All the parameters except the first are used to return the results back to
   * the caller. The caller should declare these variables and pass them here so
   * that after the execution of this method, the caller can access the results.
   *
   * @param concept_indicator_dates : Chronologically ordered observation date
   *                                  sequences for each indicator extracted
   *                                  from the JSON data in the create model
   *                                  request. This data structure is populated
   *                                  by AnalysisGraph::
   *                                  extract_concept_indicator_mapping_and_observations_from_json().
   * @param shortest_gap            : Least number of months between any two
   *                                  consecutive observations.
   * @param longest_gap             : Longest number of months between any two
   *                                  consecutive observations.
   * @param frequent_gap            : Most frequent number of months between
   *                                  two consecutive observations.
   * @param highest_frequency       : Number of time the frequent_gap is seen
   *                                  in all the observation sequences.
   * @returns void
   */
  void infer_modeling_frequency(
                        const ConceptIndicatorDates &concept_indicator_dates,
                        int &shortest_gap,
                        int &longest_gap,
                        int &frequent_gap,
                        int &highest_frequency);

  /**
   * Set the observed state sequence from the create model JSON input received
   * from the HMI.
   * The start_year, start_month, end_year, and end_month are inferred from the
   * observation sequences for indicators provided in the JSON input.
   * The sequence includes both ends of the range.
   *
   * NOTE: When Delphi is run locally, the observed state sequence is set in a
   *       separate method:
   *       AnalysisGraph::set_observed_state_sequence_from_data(), which the
   *       code could be found in train_model.cpp.
   *       It would be better if we could combine these two methods into one.
   *
   * @param json_indicators : JSON concept-indicator mapping and observations
   * @returns void
   *
   */
  void
  set_observed_state_sequence_from_json_dict(const nlohmann::json &json_indicators);

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                          create-experiment
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

  std::pair<int, int> timestamp_to_year_month(long timestamp);

  std::pair<int, int> calculate_end_year_month(int start_year, int start_month,
                                               int num_timesteps);

  double calculate_prediction_timestep_length(int start_year, int start_month,
                                              int end_year, int end_month,
                                              int pred_timesteps);

  void extract_projection_constraints(
                                const nlohmann::json &projection_constraints);

  Prediction run_causemos_projection_experiment_from_json_dict(const nlohmann::json &json_data,
                                                               int burn = 10000,
                                                               int res = 200);

  FormattedProjectionResult format_projection_result();

  void sample_transition_matrix_collection_from_prior();


  /*
   ============================================================================
   Private: Model serialization (in serialize.cpp)
   ============================================================================
  */

  void from_delphi_json_dict(const nlohmann::json &json_data, bool verbose);

  /*
   ============================================================================
   Private: Utilities (in graph_utils.cpp)
   ============================================================================
  */

  void clear_state();

  void initialize_random_number_generator();

  void remove_node(int node_id);

  // Allocate a num_verts x num_verts 2D array (std::vector of std::vectors)
  void allocate_A_beta_factors();

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

  /**
   * Utility function that converts a time range given a start date and end date
   * into an integer value.
   * At the moment returns the number of months withing the time range.
   * This should be the number of training data time points we have
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

  /*
   ============================================================================
   Private: Subgraphs (in subgraphs.cpp)
   ============================================================================
  */

  void get_subgraph(int vert,
                    std::unordered_set<int>& vertices_to_keep,
                    int cutoff,
                    bool inward);

  void get_subgraph_between(int start,
                            int end,
                            std::vector<int>& path,
                            std::unordered_set<int>& vertices_to_keep,
                            int cutoff);

  /*
   ============================================================================
   Private: Accessors
   ============================================================================
  */

  int num_nodes() { return boost::num_vertices(graph); }

  int get_vertex_id(std::string concept) {
    using namespace fmt::literals;
    try {
      return this->name_to_vertex.at(concept);
    }
    catch (const std::out_of_range& oor) {
      throw std::out_of_range("Concept \"{}\" not in CAG!"_format(concept));
    }
  }

  auto node_indices() const {
    return boost::make_iterator_range(boost::vertices(this->graph));
  };

  auto nodes() {
    using boost::adaptors::transformed;
    return this->node_indices() |
           transformed([&](int v) -> Node& { return (*this)[v]; });
  };

  auto node_names() {
    using boost::adaptors::transformed;
    return this->nodes() |
           transformed([&](auto node) -> std::string { return node.name; });
  };

  int get_degree(int vertex_id) {
    return boost::in_degree(vertex_id, this->graph) +
           boost::out_degree(vertex_id, this->graph);
  };

  auto out_edges(int i) {
    return boost::make_iterator_range(boost::out_edges(i, graph));
  }

  Node& source(EdgeDescriptor e) {
    return (*this)[boost::source(e, this->graph)];
  };

  Node& target(EdgeDescriptor e) {
    return (*this)[boost::target(e, this->graph)];
  };

  auto successors(int i) {
    return boost::make_iterator_range(boost::adjacent_vertices(i, this->graph));
  }

  auto successors(std::string node_name) {
    return this->successors(this->name_to_vertex.at(node_name));
  }

  std::vector<Node> get_successor_list(std::string node) {
    std::vector<Node> successors = {};
    for (int successor : this->successors(node)) {
      successors.push_back((*this)[successor]);
    }
    return successors;
  }

  auto predecessors(int i) {
    return boost::make_iterator_range(
        boost::inv_adjacent_vertices(i, this->graph));
  }

  auto predecessors(std::string node_name) {
    return this->predecessors(this->name_to_vertex.at(node_name));
  }

  std::vector<Node> get_predecessor_list(std::string node) {
    std::vector<Node> predecessors = {};
    for (int predecessor : this->predecessors(node)) {
      predecessors.push_back((*this)[predecessor]);
    }
    return predecessors;
  }

  double get_beta(std::string source_vertex_name,
                  std::string target_vertex_name) {

    // This is ∂target / ∂source
    return this->A_original(2 * get_vertex_id(target_vertex_name),
                            2 * get_vertex_id(source_vertex_name) + 1);
  }

  /*
   ============================================================================
   Private: Get Training Data Sequence (in train_model.cpp)
   ============================================================================
  */

  /**
   * Set the observed state sequence for a given time range from data.
   * The sequence includes both ends of the range.
   * See data.hpp::get_observations_for() for missing data rules.
   * Note: units are automatically set according
   * to the parameterization of the given CAG.
   *
   * @param start_year  : Start year of the sequence of data
   * @param start_month : Start month of the sequence of data
   * @param end_year    : End year of the sequence of data
   * @param end_month   : End month of the sequence of data
   * @param country     : Country where the data is about
   * @param state       : State where the data is about
   * @param county      : County where the data is about
   *
   */
  void
  set_observed_state_sequence_from_data(std::string country = "South Sudan",
                                        std::string state = "",
                                        std::string county = "");

  /**
   * Get the observed state (values for all the indicators)
   * for a given time point from data.
   * See data.hpp::get_observations_for() for missing data rules.
   * Note: units are automatically set according
   * to the parameterization of the given CAG.
   *
   * @param year    : Year of the time point data is extracted
   * @param month   : Month of the time point data is extracted
   * @param country : Country where the data is about
   * @param state   : State where the data is about
   * @param county  : County where the data is about
   *
   * @return        : Observed state std::vector for the specified location
   *                  on the specified time point.
   *                  Access it as: [ vertex id ][ indicator id ]
   */
  std::vector<std::vector<std::vector<double>>>
  get_observed_state_from_data(int year,
                               int month,
                               std::string country,
                               std::string state = "",
                               std::string county = "");

  /*
   ============================================================================
   Private: Initializing model parameters (in parameter_initialization.cpp)
   ============================================================================
  */

  /**
   * Initialize all the parameters and hyper-parameters of the Delphi model.
   *
   * @param start_year  : Start year of the sequence of data
   * @param start_month : Start month of the sequence of data
   * @param end_year    : End year of the sequence of data
   * @param end_month   : End month of the sequence of data
   * @param res         : Sampling resolution. The number of samples to retain.
   * @param initial_beta: Criteria to initialize β
   * @param use_heuristic : Informs how to handle missing observations.
   *                        false => let them be missing.
   *                        true => fill them. See
   *                        data.hpp::get_observations_for() for missing data
   *                        rules.
   * @param use_continuous: Choose between continuous vs discretized versions
   *                        of the differential equation solution.
   *                        Default is to use the continuous version with
   *                        matrix exponential.
   */
  void initialize_parameters(int res = 200,
                             InitialBeta initial_beta = InitialBeta::ZERO,
                             bool use_heuristic = false,
                             bool use_continuous = true);

  void set_indicator_means_and_standard_deviations();

  /**
   * To help experiment with initializing βs to different values
   *
   * @param ib: Criteria to initialize β
   */
  void init_betas_to(InitialBeta ib = InitialBeta::MEAN);

  void construct_theta_pdfs();


  /*
   ============================================================================
   Private: Training by MCMC Sampling (in sampling.cpp)
   ============================================================================
  */

  void set_base_transition_matrix();

  // Sample elements of the stochastic transition matrix from the
  // prior distribution, based on gradable adjectives.
  void set_transition_matrix_from_betas();

  void set_log_likelihood_helper(int ts);

  void set_log_likelihood();

  void set_current_latent_state(int ts);

  /**
   * Run Bayesian inference - sample from the posterior distribution.
   */
  void sample_from_posterior();

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

  /**
   * Find all the transition matrix (A) cells that are dependent on the β
   * attached to the provided edge and update them.
   * Acts upon this->A_original
   *
   * @param e: The directed edge ≡ β that has been perturbed
   */
  void update_transition_matrix_cells(EdgeDescriptor e);

  double calculate_delta_log_prior();

  void revert_back_to_previous_state();

  /*
   ============================================================================
   Private: Prediction (in prediction.cpp)
   ============================================================================
  */

  /**
   * Generate a collection of latent state sequences from the likelihood
   * model given a collection of sampled
   * (initial latent state,  transition matrix) pairs.
   *
   * @param prediction_timesteps   : The number of timesteps for the prediction
   *                                 sequences.
   * @param initial_prediction_step: The initial prediction timestep relative
   *                                 to training timesteps.
   * @param total_timesteps        : Total number of timesteps from the initial
   *                                 training date to the end prediction date.
   * @param project                : Default false. If true, generate a single
   *                                 latent state sequence based on the
   *                                 perturbed initial latent state s0.
   */
  void generate_latent_state_sequences(int initial_prediction_step);

  void perturb_predicted_latent_state_at(int timestep, int sample_number);

  /** Generate observed state sequences given predicted latent state
   * sequences using the emission model
   */
  void generate_observed_state_sequences();

  std::vector<std::vector<double>>
  generate_observed_state(Eigen::VectorXd latent_state);

  /**
   * Format the prediction result into a format Python callers favor.
   *
   * @param pred_timestes: Number of timesteps in the predicted sequence.
   *
   * @return Re-formatted prediction result.
   *         Access it as:
   *         [ sample number ][ time point ][ vertex name ][ indicator name ]
   */
  FormattedPredictionResult format_prediction_result();

  void run_model(int start_year,
                 int start_month,
                 int end_year,
                 int end_month);

  void add_constraint(int step, std::string concept_name, std::string indicator_name,
                                                double indicator_clamp_value);

  /*
   ============================================================================
   Private: Synthetic Data Experiment (in synthetic_data.cpp)
   ============================================================================
  */

  void set_random_initial_latent_state();

  void generate_synthetic_latent_state_sequence();

  void
  generate_synthetic_observed_state_sequence_from_synthetic_latent_state_sequence();

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

  /*
   ============================================================================
   Private: Graph Visualization (in graphviz.cpp)
   ============================================================================
  */

  std::pair<Agraph_t*, GVC_t*> to_agraph(
      bool simplified_labels =
          false, /** Whether to create simplified labels or not. */
      int label_depth =
          1, /** Depth in the ontology to which simplified labels extend */
      std::string node_to_highlight = "",
      std::string rankdir = "TB");

  public:
  AnalysisGraph() {
     one_off_constraints.clear();
     perpetual_constraints.clear();
  }

  ~AnalysisGraph() {}

  std::string id;
  std::string to_json_string(int indent = 0);
  bool data_heuristic = false;

  // Set the sampling resolution.
  void set_res(size_t res);

  // Get the sampling resolution.
  size_t get_res();


  /*
   ============================================================================
   Constructors from INDRA-exported JSON (in constructors.cpp)
   ============================================================================
  */

  /**
   * A method to construct an AnalysisGraph object given a JSON-serialized list
   * of INDRA statements.
   *
   * @param filename: The path to the file containing the JSON-serialized INDRA
   * statements.
   */
  static AnalysisGraph
  from_indra_statements_json_dict(nlohmann::json json_data,
                                  double belief_score_cutoff = 0.9,
                                  double grounding_score_cutoff = 0.0,
                                  std::string ontology = "WM");

  static AnalysisGraph
  from_indra_statements_json_string(std::string json_string,
                                    double belief_score_cutoff = 0.9,
                                    double grounding_score_cutoff = 0.0,
                                    std::string ontology = "WM");

  static AnalysisGraph
  from_indra_statements_json_file(std::string filename,
                                  double belief_score_cutoff = 0.9,
                                  double grounding_score_cutoff = 0.0,
                                  std::string ontology = "WM");

  /**
   * A method to construct an AnalysisGraph object given from a std::vector of
   * ( subject, object ) pairs (Statements)
   *
   * @param statements: A std::vector of CausalFragment objects
   */
  static AnalysisGraph
  from_causal_fragments(std::vector<CausalFragment> causal_fragments);

  /** From internal string representation output by to_json_string */
  static AnalysisGraph from_json_string(std::string);

  /** Copy constructor */
  AnalysisGraph(const AnalysisGraph& rhs);

  /*
   ============================================================================
   Public: Integration with Uncharted's CauseMos interface
                                                  (in causemos_integration.cpp)
   ============================================================================
  */

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            create-model
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

  /** Construct an AnalysisGraph object from JSON exported by CauseMos. */
  void from_causemos_json_dict(const nlohmann::json &json_data);

  /** Construct an AnalysisGraph object from a JSON string exported by CauseMos.
   */
  static AnalysisGraph from_causemos_json_string(std::string json_string, size_t res);

  /** Construct an AnalysisGraph object from a file containing JSON data from
   * CauseMos. */
  static AnalysisGraph from_causemos_json_file(std::string filename, size_t res);

  /**
   * Generate the response for the create model request from the HMI.
   * Calculate and return a JSON string with edge weight information for
   * visualizing AnalysisGraph models in CauseMos.
   * For now we always return success. We need to update this by conveying
   * errors into this response.
   */
  std::string generate_create_model_response();

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                          create-experiment
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

  FormattedProjectionResult
  run_causemos_projection_experiment_from_json_string(std::string json_string,
                                                      int burn = 10000,
                                                      int res = 200);

  Prediction
  run_causemos_projection_experiment_from_json_file(std::string filename,
                                                    int burn = 10000,
                                                    int res = 200);

  /*
   ============================================================================
   Private: Model serialization (in serialize.cpp)
   ============================================================================
  */

  std::string serialize_to_json_string(bool verbose = true);

  static AnalysisGraph deserialize_from_json_string(std::string json_string, bool verbose = true);

  static AnalysisGraph deserialize_from_json_file(std::string filename, bool verbose = true);

  /*
   ============================================================================
   Public: Accessors
   ============================================================================
  */

  /** Number of nodes in the graph */
  size_t num_vertices() { return boost::num_vertices(this->graph); }

  Node& operator[](std::string node_name) {
    return (*this)[this->get_vertex_id(node_name)];
  }

  Node& operator[](int v) { return this->graph[v]; }

  size_t num_edges() { return boost::num_edges(this->graph); }

  auto edges() const { return boost::make_iterator_range(boost::edges(graph)); }

  Edge& edge(EdgeDescriptor e) { return this->graph[e]; }

  Edge& edge(int source, int target) {
    return this->graph[boost::edge(source, target, this->graph).first];
  }

  Edge& edge(int source, std::string target) {
    return this->graph
        [boost::edge(source, this->get_vertex_id(target), this->graph).first];
  }

  Edge& edge(std::string source, int target) {
    return this->graph
        [boost::edge(this->get_vertex_id(source), target, this->graph).first];
  }

  Edge& edge(std::string source, std::string target) {
    return this->graph[boost::edge(this->get_vertex_id(source),
                                   this->get_vertex_id(target),
                                   this->graph)
                           .first];
  }

  boost::range_detail::integer_iterator<unsigned long> begin() {
    return boost::vertices(this->graph).first;
  };

  boost::range_detail::integer_iterator<unsigned long> end() {
    return boost::vertices(this->graph).second;
  };

  Eigen::VectorXd& get_initial_latent_state() { return this->s0; };

  /*
   ============================================================================
   Public: Graph Building (in graph_building.cpp)
   ============================================================================
  */

  void add_node(std::string concept);

  void add_edge(CausalFragment causal_fragment);
  std::pair<EdgeDescriptor, bool> add_edge(int, int);
  std::pair<EdgeDescriptor, bool> add_edge(int, std::string);
  std::pair<EdgeDescriptor, bool> add_edge(std::string, int);
  std::pair<EdgeDescriptor, bool> add_edge(std::string, std::string);

  void remove_node(std::string concept);

  // Note:
  //      Although just calling this->remove_node(concept) within the loop
  //          for( std::string concept : concept_s )
  //      is sufficient to implement this method, it is not very efficient.
  //      It re-calculates directed simple paths for each vertex removed
  //
  //      Therefore, the code in this->remove_node() has been duplicated with
  //      slightly different flow to achive a more efficient execution.
  void remove_nodes(std::unordered_set<std::string> concepts);

  void remove_edge(std::string src, std::string tgt);

  void remove_edges(std::vector<std::pair<std::string, std::string>> edges);

  /*
   ============================================================================
   Public: Subgraphs (in subgraphs.cpp)
   ============================================================================
  */

  // TODO Change the name of this function to something better, like
  // restrict_to_subgraph_for_concept, update docstring

  /**
   * Returns the subgraph of the AnalysisGraph around a concept.
   *
   * @param concept: The concept to center the subgraph about.
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

  /*
   ============================================================================
   Public: Graph Modification (in graph_modification.cpp)
   ============================================================================
  */

  void prune(int cutoff = 2);

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

  void change_polarity_of_edge(std::string source_concept,
                               int source_polarity,
                               std::string target_concept,
                               int target_polarity);

  /*
   ============================================================================
   Public: Indicator Manipulation (in indicator_manipulation.cpp)
   ============================================================================
  */

  int
  set_indicator(std::string concept, std::string indicator, std::string source);

  void delete_indicator(std::string concept, std::string indicator);

  void delete_all_indicators(std::string concept);

  /**
   * Map each concept node in the AnalysisGraph instance to one or more
   * tangible quantities, known as 'indicators'.
   *
   * @param n: Int representing number of indicators to attach per node.
   * Default is 1 since our model so far is configured for only 1 indicator per
   * node.
   */
  void map_concepts_to_indicators(int n = 1, std::string country = "");

  /*
   ============================================================================
   Public: Utilities (in graph_utils.cpp)
   ============================================================================
  */

  /*
   * Find all the simple paths between all the paris of nodes of the graph
   */
  void find_all_paths();

  void set_random_seed(int seed);

  void set_derivative(std::string, double);

  /*
   ============================================================================
   Public: Training the Model (in train_model.cpp)
   ============================================================================
  */

  /**
   * Train a prediction model given a CAG with indicators
   *
   * @param start_year  : Start year of the sequence of data
   * @param start_month : Start month of the sequence of data
   * @param end_year    : End year of the sequence of data
   * @param end_month   : End month of the sequence of data
   * @param res         : Sampling resolution. The number of samples to retain.
   * @param burn        : Number of samples to throw away. Start retaining
   *                      samples after throwing away this many samples.
   * @param country     : Country where the data is about
   * @param state       : State where the data is about
   * @param county      : county where the data is about
   * @param units       : Units for each indicator. Maps
   *                      indicator name --> unit
   * @param initial_beta: Criteria to initialize β
   * @param use_heuristic : Informs how to handle missing observations.
   *                        false => let them be missing.
   *                        true => fill them. See
   *                        data.hpp::get_observations_for() for missing data
   *                        rules.
   * @param use_continuous: Choose between continuous vs discretized versions
   *                        of the differential equation solution.
   *                        Default is to use the continuous version with
   *                        matrix exponential.
   */
  void train_model(int start_year = 2012,
                   int start_month = 1,
                   int end_year = 2017,
                   int end_month = 12,
                   int res = 200,
                   int burn = 10000,
                   std::string country = "South Sudan",
                   std::string state = "",
                   std::string county = "",
                   std::map<std::string, std::string> units = {},
                   InitialBeta initial_beta = InitialBeta::ZERO,
                   bool use_heuristic = false,
                   bool use_continuous = true);

  /*
   ============================================================================
   Public: Training by MCMC Sampling (in sampling.cpp)
   ============================================================================
  */

  void set_initial_latent_state(Eigen::VectorXd vec) { this->s0 = vec; };

  void set_default_initial_state();

  /*
   ============================================================================
   Public: Prediction (in prediction.cpp)
   ============================================================================
  */

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
   *         The first element is a std::vector of std::strings with labels for
   * each time point predicted (year-month). The second element contains
   * predicted values. Access it as: [ sample number ][ time point ][ vertex
   * name ][ indicator name ]
   */
  Prediction generate_prediction(int start_year,
                                 int start_month,
                                 int end_year,
                                 int end_month,
                                 ConstraintSchedule constraints =
                                                        ConstraintSchedule(),
                                 bool one_off = true,
                                 bool clamp_deri = true);

  /**
   * this->generate_prediction() must be called before calling this method.
   * Outputs raw predictions for a given indicator that were generated by
   * generate_prediction(). Each column is a time step and the rows are the
   * samples for that time step.
   *
   * @param indicator: A std::string representing the indicator variable for
   * which we want predictions.
   * @return A (this->res, x this->pred_timesteps) dimension 2D array
   *         (std::vector of std::vectors)
   *
   */
  std::vector<std::vector<double>> prediction_to_array(std::string indicator);

  /*
   ============================================================================
   Public: Synthetic data experiment (in synthetic_data.cpp)
   ============================================================================
  */

  std::pair<PredictedObservedStateSequence, Prediction>
  test_inference_with_synthetic_data(
      int start_year = 2015,
      int start_month = 1,
      int end_year = 2015,
      int end_month = 12,
      int res = 100,
      int burn = 900,
      std::string country = "South Sudan",
      std::string state = "",
      std::string county = "",
      std::map<std::string, std::string> units = {},
      InitialBeta initial_beta = InitialBeta::HALF,
      bool use_continuous = true);

  /*
   ============================================================================
   Public: Graph Visualization (in graphviz.cpp)
   ============================================================================
  */

  std::string to_dot();

  void
  to_png(std::string filename = "CAG.png",
         bool simplified_labels =
             false, /** Whether to create simplified labels or not. */
         int label_depth =
             1, /** Depth in the ontology to which simplified labels extend */
         std::string node_to_highlight = "",
         std::string rankdir = "TB");

  /*
   ============================================================================
   Public: Printing (in printing.cpp)
   ============================================================================
  */

  void print_nodes();
  void print_edges();
  void print_name_to_vertex();
  void print_indicators();
  void print_A_beta_factors();
  void print_latent_state(const Eigen::VectorXd&);

  /*
   * Prints the simple paths found between all pairs of nodes of the graph
   * Groupd according to the starting and ending vertex.
   * find_all_paths() should be called before this to populate the paths
   */
  void print_all_paths();

  // Given an edge (source, target vertex ids - i.e. a β ≡ ∂target/∂source),
  // print all the transition matrix cells that are dependent on it.
  void print_cells_affected_by_beta(int source, int target);

  void print_training_range();
};
