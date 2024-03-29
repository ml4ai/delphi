#pragma once

#include "definitions.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <boost/graph/graph_traits.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/iterator_range.hpp>
#include <range/v3/all.hpp>

#include <sqlite3.h>

#include "graphviz_interface.hpp"

#include "DiGraph.hpp"
#include "Tran_Mat_Cell.hpp"
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#ifdef MULTI_THREADING
  #include <future>
#endif

#ifdef TIME
  #include "CSVWriter.hpp"
#endif

const double tuning_param = 1.0;

enum InitialBeta { ZERO, ONE, HALF, MEAN, MEDIAN, PRIOR, RANDOM };
//enum class InitialBeta : char { ZERO, ONE, HALF, MEAN, MEDIAN, PRIOR, RANDOM };
enum InitialDerivative { DERI_ZERO, DERI_PRIOR };
enum HeadNodeModel { HNM_NAIVE, HNM_FOURIER };

typedef std::unordered_map<std::string, std::vector<double>>
    AdjectiveResponseMap;

// This is a multimap to keep provision to have multiple observations per
// time point per indicator.
// Access (concept is a vertex in the CAG)
// [ concept ][ indicator ][ epoch --→ observation ]
typedef std::vector<std::vector<std::multimap<long, double>>>
    ConceptIndicatorData;

// Keeps the sequence of dates for which data points are available
// Data points are sorted according to dates
// Access:
// [ concept ][ indicator ][epoch]
typedef std::vector<std::vector<long>> ConceptIndicatorEpochs;

// Access
// [ timestep ][ concept ][ indicator ][ observation ]
typedef std::vector<std::vector<std::vector<std::vector<double>>>>
    ObservedStateSequence;

typedef std::vector<std::vector<std::vector<double>>>
    PredictedObservedStateSequence;

typedef std::pair<std::tuple<std::string, int, std::string>,
                  std::tuple<std::string, int, std::string>>
    CausalFragment;

// { concept_name --> (ind_name, [obs_0, obs_1, ... ])}
typedef std::unordered_map<std::string,
                           std::pair<std::string, std::vector<double>>>
    ConceptIndicatorAlignedData;

typedef std::tuple<std::vector<std::string>, std::vector<int>, std::string>
    EventCollection;

typedef std::pair<EventCollection, EventCollection>
    CausalFragmentCollection;

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

// Format AnalysisGraph state to output
// [ concept name ] --> [ ind1, ind2, ... ]
typedef std::unordered_map<std::string, std::vector<std::string>> ConceptIndicators;

// List of edges [(source, target), ...]
typedef std::vector<std::pair<std::string, std::string>> Edges;

// List of adjectives [(source, target), ...]
typedef std::vector<std::pair<std::string, std::string>> Adjectives;

// List of polarities [(source, target), ...]
typedef std::vector<std::pair<int, int>> Polarities;

// Vector of theta priors and samples pairs for each edge
// Ordering is according to order of edges in Edges data vector
// For each edge, there is a tuple of vectors
// first element of the tuple is a vector of theta priors KDEs
// second element of the tuple is a vector of sampled thetas
// [([p1, p1, ...], [s1, s2, ...]), ... ]
// Each tuple: <dataset, sampled thertas, log prior histogram>
// TODO: remove dataset and convert this to a pair
//typedef std::vector<std::pair<std::vector<double>, std::vector<double>>> Thetas;
typedef std::vector<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>> Thetas;

// Sampled Derivatives for each concept
// Access
// [ concept name ] --> [s_1, s_2, ..., s_res ]
typedef std::unordered_map<std::string, std::vector<double>> Derivatives;

// Data
// Access
// [ indicator name ] --> {
//                           [ "Time Step" ] --> [ts1, ts2, ...]
//                           [ "Data" ]      --> [ d1,  d2, ...]
//                        }
typedef std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> Data;

// Predictions
// Access
// [ indicator name ] --> {
//                           [ ts ] --> [p_1, p_2, p_3, ..., p_res]
//                        }
typedef std::unordered_map<std::string, std::unordered_map<int, std::vector<double>>> Predictions;

typedef std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>>
    CredibleIntervals;

typedef std::tuple<
    ConceptIndicators,
    Edges, // List of edges [(source, target), ...]
    Adjectives,
    Polarities,
    // Theta priors and samples for each edge
    // [(priors, samples), ... ]
    Thetas,
    Derivatives,
    // Data year month range
    //std::vector<std::string>,
    std::vector<long>,
    // Data
    Data,
    // Prediction year month range
    //std::vector<std::string>,
    std::vector<double>,
    Predictions,
    CredibleIntervals,
    // Log likelihoods
    std::vector<double>,
    int // Number of bins in theta prior distributions
            > CompleteState;

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
  #ifdef TIME
    std::pair<std::vector<std::string>, std::vector<long>> durations;
    std::pair<std::vector<std::string>, std::vector<long>> mcmc_part_duration;
    CSVWriter writer;
    std::string timing_file_prefix = "";
    int timing_run_number = 0;
  #endif

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

  // Uniform discrete distributions used by the MCMC sampler
  // to perturb the initial latent state
  std::uniform_int_distribution<int> uni_disc_dist;
  // to sample an edge
  std::uniform_int_distribution<int> uni_disc_dist_edge;

  // Sampling resolution
  size_t res;

  // Number of KDE kernels
  size_t n_kde_kernels = 200;

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

  std::unordered_set<int> body_nodes = {};
  std::unordered_set<int> head_nodes = {};
  std::vector<double> generated_latent_sequence;
  int generated_concept;

  /*
   ============================================================================
   Sampler Related Variables
   ============================================================================
  */

  // Used to check whether there is a trained model before calling
  // generate_prediction()
  bool trained = false;

  // training was stopped by user input
  bool stopped = false;

  int n_timesteps = 0;
  int pred_timesteps = 0;
  std::pair<std::pair<int, int>, std::pair<int, int>> training_range;
  std::vector<std::string> pred_range;
  long train_start_epoch = -1;
  long train_end_epoch = -1;
  double pred_start_timestep = -1;
  // Number of zero based observation timesteps to each observation point.
  // When data aggregation level is MONTHLY, this is the number of months
  // When data aggregation level is YEARLY, this is the number of years
  std::vector<long> observation_timesteps_sorted;
  std::vector<double> modeling_timestep_gaps;
  std::vector<double> observation_timestep_unique_gaps;
  // Currently Delphi could work either with monthly data or yearly data
  // but not with some monthly and some yearly
  DataAggregationLevel model_data_agg_level = DataAggregationLevel::MONTHLY;

  #ifdef MULTI_THREADING
    // A future cannot be copied. So we need to specify a copy assign
    // constructor that does not copy this for the code to compile
    std::vector<std::future<Eigen::MatrixXd>> matrix_exponential_futures;
  #endif
  std::unordered_map<double, Eigen::MatrixXd> e_A_ts;
  std::unordered_map<double, Eigen::MatrixXd> previous_e_A_ts;

  // Computed matrix exponentials of A_fourier_base:
  // e^(A_fourier_base * gap)
  // for all the gaps we have to advance the system to evolve it to the time
  // steps where there are observations.
  // Access: [gap] --> e^(A_fourier_base * gap)
  std::unordered_map<double, Eigen::MatrixXd> e_A_fourier_ts;

  long num_modeling_timesteps_per_one_observation_timestep = 1;

  std::unordered_map<int, std::function<double(unsigned int, double)>> external_concepts;
  std::vector<unsigned int> concept_sample_pool;
  std::vector<EdgeDescriptor> edge_sample_pool;

  double t = 0.0;
  double delta_t = 1.0;

  double log_likelihood = 0.0;
  double previous_log_likelihood = 0.0;
  double log_likelihood_MAP = 0.0;
  int MAP_sample_number = -1;
  std::vector<double> log_likelihoods;

  // To decide whether to perturb a θ or a derivative
  // If coin_flip < coin_flip_thresh perturb θ else perturb derivative
  double coin_flip = 0;
  double coin_flip_thresh = 0.5;

  // Remember the old θ, logpdf(θ) and the edge where we perturbed the θ.
  // We need this to revert the system to the previous state if the proposal
  // gets rejected.
  // Access:
  //        edge, θ, logpdf(θ)
  std::tuple<EdgeDescriptor, double, double> previous_theta;

  // Remember the old derivative and the concept we perturbed the derivative
  int changed_derivative = 0;
  double previous_derivative = 0;

  // Latent state that is evolved by sampling.
  Eigen::VectorXd s0;
  Eigen::VectorXd s0_prev;
  double derivative_prior_variance = 0.1;

  // Transition matrix that is evolved by sampling.
  // Since variable A has been already used locally in other methods,
  // I chose to name this A_original. After refactoring the code, we could
  // rename this to A.
  Eigen::MatrixXd A_original;
  Eigen::MatrixXd previous_A_original;

  HeadNodeModel head_node_model = HeadNodeModel::HNM_NAIVE;//HeadNodeModel::HNM_FOURIER;//
  // Base transition matrix for the Fourier decomposition based head node model
  Eigen::MatrixXd A_fourier_base;
  // Initial state for the Fourier decomposition based head node model
  Eigen::VectorXd s0_fourier;

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
  std::unordered_map<int, std::vector<std::pair<int, double>>>
      head_node_one_off_constraints;
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
  //std::vector<std::vector<double>> latent_mean_collection;
  //std::vector<std::vector<double>> latent_std_collection;
  // Access:
  // [sample][node id]{partition --> (mean, std)}
  //std::vector<std::vector<std::unordered_map<int, std::pair<double, double>>>> latent_mean_std_collection;

  std::vector<Eigen::VectorXd> synthetic_latent_state_sequence;
  bool synthetic_data_experiment = false;

  /*
   ============================================================================
   Private: Integration with Uncharted's CauseMos interface
                                                  (in causemos_integration.cpp)
   ============================================================================
  */

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            training-progress
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

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
   * @param concept_indicator_epochs : This data structure gets filled with
   *                                  epochs where observations
   *                                  are available for each indicator. Each
   *                                  indicator gets a separate sequence of
   *                                  chronologically ordered epochs.
   *                                  These are used to asses the best
   *                                  frequency to align observations across
   *                                  all the indicators.
   * @returns void
   *
   */
  void extract_concept_indicator_mapping_and_observations_from_json(
                        const nlohmann::json &json_indicators,
                        ConceptIndicatorData &concept_indicator_data,
                        ConceptIndicatorEpochs &concept_indicator_epochs);


  static double epoch_to_timestep(long epoch, long train_start_epoch, long modeling_frequency);

  /** Infer the best sampling period to align observations to be used as the
   * modeling frequency from all the observation sequences.
   *
   * We consider the sequence of epochs where observations are available and
   * then the gaps in epochs between adjacent observations. We take the most
   * frequent gap as the modeling frequency. When more than one gap is most
   * frequent, we take the smallest such gap.
   *
   * NOTE: Some thought about how to use this information:
   * shortest_gap = longest_gap  ⇒ no missing data
   * shortest_gap < longest_gap ⇒ missing data
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
   * @param concept_indicator_observation_timesteps : Chronologically ordered observation epoch
   *                                  sequences for each indicator extracted
   *                                  from the JSON data in the create model
   *                                  request. This data structure is populated
   *                                  by AnalysisGraph::
   *                                  extract_concept_indicator_mapping_and_observations_from_json().
   * @param shortest_gap            : Least number of epochs between any two
   *                                  consecutive observations.
   * @param longest_gap             : Most number of epochs between any two
   *                                  consecutive observations.
   * @param frequent_gap            : Most frequent number of epochs between
   *                                  two consecutive observations.
   * @param highest_frequency       : Number of time the frequent_gap is seen
   *                                  in all the observation sequences.
   * @returns epochs_sorted         : A sorted list of epochs where observations
   *                                  are present for at least one indicator
   */
  void infer_modeling_period(
                        const ConceptIndicatorEpochs & concept_indicator_observation_timesteps,
                        long &shortest_gap,
                        long &longest_gap,
                        long &frequent_gap,
                        int &highest_frequency);

  void infer_concept_period(const ConceptIndicatorEpochs &concept_indicator_epochs);

  /**
   * Set the observed state sequence from the create model JSON input received
   * from the HMI.
   * The training_start_epoch and training_end_epochs are extracted from the
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

  // Epoch --> (year, month, date)
  std::tuple<int, int, int> epoch_to_year_month_date(long epoch);

  void extract_projection_constraints(
                                const nlohmann::json &projection_constraints, long skip_steps);

  FormattedProjectionResult run_causemos_projection_experiment_from_json_dict(
                                               const nlohmann::json &json_data);

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
                             InitialDerivative initial_derivative = InitialDerivative::DERI_ZERO,
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

  #ifdef MULTI_THREADING
    void compute_multiple_matrix_exponentials_parallelly(
                         const Eigen::MatrixXd & A,
                         std::vector<std::future<Eigen::MatrixXd>> &me_futures);
  #endif

  // Sample elements of the stochastic transition matrix from the
  // prior distribution, based on gradable adjectives.
  void set_transition_matrix_from_betas();

  void set_log_likelihood_helper(int ts);

  void set_log_likelihood();

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
   Private: Modeling head nodes (in head_nodes.cpp)
   ============================================================================
  */

  void partition_data_and_calculate_mean_std_for_each_partition(
      Node& n, std::vector<double>& latent_sequence);

  void apply_constraint_at(int ts, int node_id);

  void generate_head_node_latent_sequence(int node_id,
                                          int num_timesteps,
                                          bool sample,
                                          int seq_no = 0);

  void generate_head_node_latent_sequence_from_changes(Node &n,
                                                       int num_timesteps,
                                                       bool sample);

  void generate_head_node_latent_sequences(int samp, int num_timesteps);

  void update_head_node_latent_state_with_generated_derivatives(
      int ts_current,
      int ts_next,
      int concept_id,
      std::vector<double>& latent_sequence);

  void update_latent_state_with_generated_derivatives(int ts_current,
                                                      int ts_next);

  /*
   ============================================================================
   Private: Prediction (in prediction.cpp)
   ============================================================================
  */

  /**
   * Check whether each latent state value is within the minimum and maximum
   * range of values for the first indicator attached to each latent node, if
   * such bounds are provided for that indicator.
   *
   * Latent state bounds are computed by dividing the observation bound by the
   * scaling factor of the first indicator attached to that node.
   */
  void check_bounds();


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
  void generate_latent_state_sequences(double initial_prediction_step);

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

  /*
   ============================================================================
   Private: Fourier decomposition based seasonal head node model (in fourier.cpp)
   ============================================================================
  */

  /**
   * Generates a vector of all the effective frequencies for a particular period
   * @param n_components: The number of pure sinusoidal frequencies to generate
   * @param period: The period of the variable(s) being modeled by Fourier
   *                decomposition. All variable(s) share the same period.
   * @return A vector of all the effective frequencies
   */
  std::vector<double> generate_frequencies_for_period(int period,
                                                      int n_components);

  /**
   * Evolves the provided LDS (A_base and _s0) n_modeling_time_steps (e.g.
   * months) taking step_size steps each time. For example, if the step_size is
   * 0.25, there will be 4 prediction points per one modeling time step.
   * @param A_base: Base transition matrix that define the LDS.
   * @param _s0: Initial state of the system. Used _s0 instead of s0 because s0
   *             is the member variable that represent the initial state of the
   *             final system that is used by the AnalysisGraph object.
   * @param n_modeling_time_steps: The number of modeling time steps (full time
   *                               steps, e.g. months) to evolve the system.
   * @param step_size: Amount to advance the system at each step.
   * @return A matrix of evolved values. Each column has values for one step.
   *              row 2i   - Values for variable i in the system
   *              row 2i+1 - Derivatives for variable i in the system
   */
  Eigen::MatrixXd evolve_LDS(const Eigen::MatrixXd &A_base,
                             const Eigen::VectorXd &_s0,
                             int n_modeling_time_steps,
                             double step_size = 1);

  /**
   * Assemble the LDS to generate sinusoidals of desired effective frequencies
   * @param freqs: A vector of effective frequencies
   *               (λω; ω = 1, 2, ... & λ = 2π/period)
   * @return pair (base transition matrix, initial state)
   *         0 radians is the initial angle.
   */
  std::pair<Eigen::MatrixXd, Eigen::VectorXd>
    assemble_sinusoidal_generating_LDS(const std::vector<double> &freqs);

  /*
  std::pair<Eigen::MatrixXd, Eigen::VectorXd>
    assemble_sinusoidal_generating_LDS(unsigned short components,
                                       unsigned short period);
  */

  /**
   * Generate a matrix of sinusoidal values of all the desired effective
   * frequencies for all the bin locations.
   * @param A_sin_base: Base transition matrix for sinusoidal generating LDS
   * @param s0_sin: Initial state (0 radians) for sinusoidal generating LDS
   * @param period: Period shared by the time series that will be fitted using
   *                the generated sinusoidal values. This period must be the
   *                same period used to generate the vector of effective
   *                frequencies used to assemble the transition matrix
   *                A_sin_base and the initial state s0_sin.
   * @return A matrix of required sinusoidal values.
   *         row t contains sinusoidals for bin t (radians)
   *         col 2ω contains sin(λω t)
   *         col 2ω+1 contains λω cos(λω t)
   *
   *         with ω = 1, 2, ... & λ = 2π/period
   */
  Eigen::MatrixXd generate_sinusoidal_values_for_bins(const Eigen::MatrixXd &A_sin_base,
                                                      const Eigen::VectorXd &s0_sin,
                                                      int period);

  // NOTE: This method could be made a method of the Node class. The best
  //       architecture would be to make a subclass, HeadNode, of Node class and
  //       include this method there. At the moment we incrementally create the
  //       graph while identifying head nodes, we are using Node objects
  //       everywhere. To follow the HeadNode subclass specialization route, we
  //       either have to replace Node objects with HeadNode objects or do a
  //       first pass through the input to identify head nodes and then create
  //        the graph.
  /**
   * For each head node, computes the Fourier coefficients to fit a seasonal
   * curve to partitioned observations (bins) using the least square
   * optimization.
   * @param sinusoidals: Sinusoidal values of required effective frequencies at
   *                     each bin position. Row b contains all the sinusoidal
   *                     values for bin b.
   *                        sinusoidals(b, 2(ω-1))     =    sin(λω b)
   *                        sinusoidals(b, 2(ω-1) + 1) = λω cos(λω b)
   *                    with ω = 1, 2, ... & λ = 2π/period
   * @param n_components: The number of different sinusoidal frequencies used to
   *                      fit the seasonal head node model. The supplied
   *                      sinusoidals matrix could have sinusoidal values for
   *                      higher frequencies than needed
   *                      (number of columns > 2 * n_components). This method
   *                      utilizes only the first 2 * n_components columns.
   * @param head_node_ids: A list of head nodes with the period matching the
   *                       period represented in the provided sinusoidals. The
   *                       period of all the head nodes in this list must be the
   *                       same as the period parameter used when generating the
   *                       sinusoidals.
   * @return The Fourier coefficients in the order: α₀, β₁, α₁, β₂, α₂, ...
   *         α₀ is the coefficient for    cos(0)/2  term
   *         αᵢ is the coefficient for λi cos(λi b) term
   *         βᵢ is the coefficient for    sin(λi b) term
   *
   *         with i = 1, 2, ... & λ = 2π/period & b = 0, 1, ..., period - 1
   */
  void compute_fourier_coefficients_from_least_square_optimization(
                                            const Eigen::MatrixXd &sinusoidals,
                                            int n_components,
                                            std::vector<int> &head_node_ids);

  /**
   * Assembles the LDS to generate the head nodes specified in the hn_to_mat_row
   * map by Fourier reconstruction using the sinusoidal frequencies generated by
   * the provided sinusoidal generating LDS: A_sin_base and s0_sin.
   * @param A_sin_base: Base transition matrix to generate sinusoidals with all
   *                    the possible effective frequencies.
   * @param s0_sin: Initial state of the LDS that generates sinusoidal curves.
   * @param n_components: The number of different sinusoidal frequencies used to
   *                      fit the seasonal head nodes. The sinusoidal generating
   *                      LDS provided (A_sin_base, s0_sin) could generate more
   *                      sinusoidals of higher frequencies. This method uses
   *                      only the lowest n_components frequencies to assemble
   *                      the complete system.
   * @param n_concepts: The number of concepts being modeled by thi LDS. For the
   *                    LDS to correctly include all the seasonal head nodes
   *                    specified in hn_to_mat_row:
   *                    n_concepts ≥ hn_to_mat_row.size()
   * @param hn_to_mat_row: A map that maps each head node being modeled to the
   *                       transition matrix rows and state vector rows.
   *                       Each concept is allocated to two consecutive rows.
   *                       In the transition matrix:
   *                             even row) for first derivative and
   *                             odd  row) for second derivative
   *                       In the state vector:
   *                             even row) the value
   *                             odd  row) for first derivative and
   *                       This map indicates the even row numbers each concept
   *                       is assigned to.
   * @return pair (base transition matrix, initial state) for the complete LDS
   *         with the specified number of sinusoidal frequencies (n_components)
   *         0 radians is the initial angle.
   */
  std::pair<Eigen::MatrixXd, Eigen::VectorXd>
  assemble_head_node_modeling_LDS(const Eigen::MatrixXd &A_sin_base,
                                  const Eigen::VectorXd &s0_sin,
                                  int n_components,
                                  int n_concepts,
                                  std::unordered_map<int, int> &hn_to_mat_row);

  /**
   * Evolves the Fourier decomposition based seasonal head node model assembled
   * for head nodes that share the same period for one period at between bin
   * midpoints. Then computes the variable wise root mean squared error for the
   * predictions against binned data, notes down the parameters when any rmse
   * reduces. The parameter n_components should be the same number of sinusoidal
   * frequencies used when assembling the supplied Fourier decomposition based
   * seasonal head node model LDS (A_hn_period_base and s0_hn_period).
   * @param A_hn_period_base: Transition matrix for the LDS that models seasonal
   *                          head nodes with the same period.
   * @param s0_hn_period: Initial state for the LDS modeling seasonal head nodes
   *                      with the same period. t₀ = 0 radians.
   * @param period: Period of the seasonal head nodes modeled by the LDS defined
   *                by A_hn_period_base and s0_hn_period.
   * @param n_components: Total number of sinusoidal frequencies used to model
   *                      all the seasonal head nodes in this LDS.
   * @param hn_to_mat_row: A map that maps each head node being modeled to the
   *                       transition matrix rows and state vector rows.
   *                       Each concept is allocated to two consecutive rows.
   *                       In the transition matrix:
   *                             even row) for first derivative
   *                             odd  row) for second derivative
   *                       In the state vector:
   *                             even row) the value
   *                             odd  row) for first derivative
   *                       This map indicates the even row numbers each concept
   *                       is assigned to.
   * @return A boolean value indicating whether the Fourier decomposition based
   *         seasonal head node model got improved for any of the head nodes
   *         specified in hn_to_mat_row.
   *              true  ⇒ Got improved. Should check for n_components + 1
   *              false ⇒ Did not improve. No point in checking for
   *                      n_components + 1. Stop early.
   */
  bool determine_the_best_number_of_components(
                                   const Eigen::MatrixXd & A_hn_period_base,
                                   const Eigen::VectorXd & s0_hn_period,
                                   int period,
                                   int n_components,
                                   std::unordered_map<int, int> &hn_to_mat_row);

  /**
   * Assembles the complete LDS for all the seasonal head nodes combining the
   * best Fourier decomposition based seasonal model for each head node.
   * @param fourier_frequency_set: A set of all the sinusoidal frequencies
   *                               needed to model all the seasonal head nodes.
   * @param n_concepts: The number of concepts being modeled by thi LDS. For the
   *                    LDS to correctly include all the seasonal head nodes
   *                    specified in hn_to_mat_row:
   *                    n_concepts ≥ hn_to_mat_row.size()
   * @param hn_to_mat_row: A map that maps each head node being modeled to the
   *                       transition matrix rows and state vector rows.
   *                       Each concept is allocated to two consecutive rows.
   *                       In the transition matrix:
   *                             even row) for first derivative
   *                             odd  row) for second derivative
   *                       In the state vector:
   *                             even row) the value
   *                             odd  row) for first derivative and
   *                       This map indicates the even row numbers each concept
   *                       is assigned to.
   * @return The final LDS that models all the seasonal head nodes. Pairs of
   *         zero rows are left where the LDS would be modeling body nodes. For
   *         the transition matrix to be completed, the top left 2 * n_concepts
   *         by 2 * n_concepts square black of the returned transition matrix
   *         should be filled according to the relationships specified by the
   *         CAG. The respective rows of the initial state should also be filled
   *         accordingly.
   */
  std::pair<Eigen::MatrixXd, Eigen::VectorXd>
      assemble_all_seasonal_head_node_modeling_LDS(
                               std::unordered_set<double> fourier_frequency_set,
                               int n_concepts,
                               std::unordered_map<int, int> &hn_to_mat_row);

  void assemble_base_LDS(InitialDerivative id);

  /**
   * Merges the LDS that defines relationships between concepts into the Fourier
   * decomposition based seasonal head node model LDS.
   * The transition matrix of the concept LDS is inserted as the first block
   * along the diagonal of the seasonal head node model transition matrix.
   * The initial states of the concept LDS and the seasonal head node models are
   * merged such that the head node states are taken from the seasonal head node
   * model initial state and the body node states are taken from the initial
   * state of the concept LDS.
   * The initial state merging math counts on that seasonal head node model
   * initial state has zeros for body node state and the concept LDS initial
   * state has zeros for head node state. This way, we could sum the two states
   * to combine them.
   * The merged LDS becomes available in the two member variables:
   * A_fourier_base and current_latent_state.
   * @param A_concept_base: The base transition matrix of the concept LDS
   * @param s0_concept: The initial states of the concept LDS
   */
  void merge_concept_LDS_into_seasonal_head_node_modeling_LDS(
                                          const Eigen::MatrixXd &A_concept_base,
                                          const Eigen::VectorXd s0_concept);

  /**
   * Evolves the provided LDS (A_base and _s0) for n_time_steps modeling time
   * steps and outputs the prediction matrix to a csv file:
   *      col 2i   - Predictions for variable i in the system
   *      col 2i+1 - Derivatives for variable i in the system
   *      Each row is a time step
   * Predicts four steps for each modeling time step.
   * @param A_base: Base transition matrix that define the LDS.
   * @param _s0: Initial state of the system (t₀ = 0 radians). Used _s0 instead of
   *             s0 because s0 is the member variable that represent the initial
   *             state of the final system that is used by the AnalysisGraph
   *             object.
   * @param n_time_steps: The number of modeling time steps (full time steps,
   *                      e.g. months) to evolve the system.
   */
  void predictions_to_csv(const Eigen::MatrixXd &A_base,
                          const Eigen::VectorXd &_s0, int n_time_steps);

  /**
   * The main driver method that fits the Fourier decomposition based seasonal
   * model to all the seasonal head nodes.
   * @return The final LDS that models all the seasonal head nodes combining the
   *         best Fourier decomposition based seasonal model for each head node.
   *         Pairs of zero rows are left where the LDS would be modeling body
   *         bodes. For the transition matrix to be completed, the top left
   *         2 * n_concepts by 2 * n_concepts square black of the returned
   *         transition matrix should be filled according to the relationships
   *         specified by the CAG.
   *         The respective rows of the initial state should also be filled
   *         accordingly.
   */
  std::pair<Eigen::MatrixXd, Eigen::VectorXd>
  fit_seasonal_head_node_model_via_fourier_decomposition();

  public:
  AnalysisGraph() {
     one_off_constraints.clear();
     perpetual_constraints.clear();
  }

  ~AnalysisGraph() {}

  std::string id;
  std::string experiment_id = "experiment_id_not_set";

  std::string to_json_string(int indent = 0);
  bool data_heuristic = false;

  // Set the sampling resolution.
  void set_res(size_t res);

  // Set the number of KDE kernels.
  void set_n_kde_kernels(size_t kde_kernels)
      {this->n_kde_kernels = kde_kernels;};

  // Get the sampling resolution.
  size_t get_res();


  // there may be a better place in this file for this prototype
  /** Construct an AnalysisGraph object from JSON exported by CauseMos. */
  void from_causemos_json_dict(const nlohmann::json &json_data,
                               double belief_score_cutoff,
                               double grounding_score_cutoff);

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

  static AnalysisGraph
  from_causal_fragments_with_data(std::pair<std::vector<CausalFragment>,
                                  ConceptIndicatorAlignedData> cag_ind_data,
                                  int kde_kernels = 5);

  /** From internal string representation output by to_json_string */
  static AnalysisGraph from_json_string(std::string);

  /** Copy constructor */
  AnalysisGraph(const AnalysisGraph& rhs);

  /** Copy assignment operator */
  AnalysisGraph& operator=(AnalysisGraph rhs);

  /*
   ============================================================================
   Public: Integration with Uncharted's CauseMos interface
                                                  (in causemos_integration.cpp)
   ============================================================================
  */

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            training-progress
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  bool get_trained(){ return trained; }
  bool get_stopped() { return stopped; }
  double get_log_likelihood(){ return log_likelihood; }
  double get_previous_log_likelihood(){ return previous_log_likelihood; }
  double get_log_likelihood_MAP(){ return log_likelihood_MAP; }

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            create-model
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

  /** Construct an AnalysisGraph object from a JSON string exported by CauseMos.
   */
  static AnalysisGraph from_causemos_json_string(std::string json_string,
                                                 double belief_score_cutoff = 0,
                                                 double grounding_score_cutoff = 0,
                                                 int kde_kernels = 4);

  /** Construct an AnalysisGraph object from a file containing JSON data from
   * CauseMos. */
  static AnalysisGraph from_causemos_json_file(std::string filename,
                                               double belief_score_cutoff = 0,
                                               double grounding_score_cutoff = 0,
                                               int kde_kernels = 4);

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
  run_causemos_projection_experiment_from_json_string(std::string json_string);

  FormattedProjectionResult
  run_causemos_projection_experiment_from_json_file(std::string filename);


    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                      edit-weights
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

  /**
   *
   * @param source Source concept name
   * @param target Target concept name
   * @param scaled_weight A value in the range [0, 1]. Delphi edge weights are
   *               angles in the range [-π/2, π/2]. Values in the range ]0, π/2[
   *               represents positive polarities and values in the range
   *               ]-π/2, 0[ represents negative polarities.
   * @param polarity Polarity of the edge. Should be either 1 or -1.
   * @return 0 freezing the edge is successful
   *         1 scaled_weight outside accepted range
   *         2 Source concept does not exist
   *         4 Target concept does not exist
   *         8 Edge does not exist
   */
  unsigned short freeze_edge_weight(std::string source, std::string target,
                          double scaled_weight, int polarity);


  /*
   ============================================================================
   Public: Model serialization (in serialize.cpp)
   ============================================================================
  */

  std::string serialize_to_json_string(bool verbose = true, bool compact = true);

  void export_create_model_json_string();

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

  double get_MAP_log_likelihood() { return this->log_likelihood_MAP; };

  /*
   ============================================================================
   Public: Graph Building (in graph_building.cpp)
   ============================================================================
  */

  int add_node(std::string concept);

  bool add_edge(CausalFragment causal_fragment);
  void add_edge(CausalFragmentCollection causal_fragments);
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
                   InitialDerivative initial_derivative = InitialDerivative::DERI_ZERO,
                   bool use_heuristic = false,
                   bool use_continuous = true);

  void run_train_model(int res = 200,
                   int burn = 10000,
                   HeadNodeModel head_node_model = HeadNodeModel::HNM_NAIVE,
                   InitialBeta initial_beta = InitialBeta::ZERO,
                   InitialDerivative initial_derivative = InitialDerivative::DERI_ZERO,
                   bool use_heuristic = false,
                   bool use_continuous = true,
                   int train_start_timestep = 0,
                   int train_timesteps = -1,
                   std::unordered_map<std::string, int> concept_periods = {},
                   std::unordered_map<std::string, std::string> concept_center_measures = {},
                   std::unordered_map<std::string, std::string> concept_models = {},
                   std::unordered_map<std::string, double> concept_min_vals = {},
                   std::unordered_map<std::string, double> concept_max_vals = {},
                   std::unordered_map
                       <std::string, std::function<double(unsigned int, double)>>
                   ext_concepts = {});

  void run_train_model_2(int res = 200,
                       int burn = 10000,
                       InitialBeta initial_beta = InitialBeta::ZERO,
                       InitialDerivative initial_derivative = InitialDerivative::DERI_ZERO,
                       bool use_heuristic = false,
                       bool use_continuous = true);

  /*
   ============================================================================
   Public: Training by MCMC Sampling (in sampling.cpp)
   ============================================================================
  */

  void set_initial_latent_state(Eigen::VectorXd vec) { this->s0 = vec; };

  void set_default_initial_state(InitialDerivative id = InitialDerivative::DERI_ZERO);

  static void check_multithreading();
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

  void generate_prediction(int pred_start_timestep,
                           int pred_timesteps,
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

  static AnalysisGraph generate_random_CAG(unsigned int num_nodes,
                                           unsigned int num_extra_edges = 0);

  void generate_synthetic_data(unsigned int num_obs = 48,
                               double noise_variance = 0.1,
                               unsigned int kde_kernels = 1000,
                               InitialBeta initial_beta = InitialBeta::PRIOR,
                               InitialDerivative initial_derivative = InitialDerivative::DERI_PRIOR,
                               bool use_continuous = false);

  void initialize_random_CAG(unsigned int num_obs,
                             unsigned int kde_kernels,
                             InitialBeta initial_beta,
                             InitialDerivative initial_derivative,
                             bool use_continuous);

  void interpolate_missing_months(std::vector<int> &filled_months, Node &n);

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

  void print_MAP_estimate();

  /*
   ============================================================================
   Public: Formatting output (in format_output.cpp)
   ============================================================================
  */

  CredibleIntervals get_credible_interval(Predictions preds);

  CompleteState get_complete_state();

  /*
   ============================================================================
   Public: Database interactions (in database.cpp)
   ============================================================================
  */

  sqlite3* open_delphi_db(int mode = SQLITE_OPEN_READONLY);

  void write_model_to_db(std::string model_id);

  AdjectiveResponseMap construct_adjective_response_map(
      std::mt19937 gen,
      std::uniform_real_distribution<double>& uni_dist,
      std::normal_distribution<double>& norm_dist,
      size_t n_kernels
  );

  /*
   ============================================================================
   Public: Profiling Delphi (in profiler.cpp)
   ============================================================================
  */

  void initialize_profiler(int res = 100,
                           int kde_kernels = 1000,
                           InitialBeta initial_beta = InitialBeta::ZERO,
                           InitialDerivative initial_derivative = InitialDerivative::DERI_ZERO,
                           bool use_continuous = true);

  void profile_mcmc(int run = 1, std::string file_name_prefix = "mcmc_timing");

  void profile_kde(int run = 1, std::string file_name_prefix = "kde_timing");

  void profile_prediction(int run = 1, int pred_timesteps = 24, std::string file_name_prefix = "prediction_timing");

  void profile_matrix_exponential(int run = 1,
                                  std::string file_name_prefix = "mat_exp_timing",
                                  std::vector<double> unique_gaps = {1, 2, 5},
                                  int repeat = 30,
                                  bool multi_threaded = false);

#ifdef TIME
  void set_timing_file_prefix(std::string tfp) {this->timing_file_prefix = tfp;}
  void create_mcmc_part_timing_file()
  {
      std::string filename = this->timing_file_prefix + "embeded_" +
                              std::to_string(this->num_nodes()) + "-" +
                              std::to_string(this->num_nodes()) + "_" +
                              std::to_string(this->timing_run_number) + "_" +
                              delphi::utils::get_timestamp() + ".csv";
      this->writer = CSVWriter(filename);
      std::vector<std::string> headings = {"Run", "Nodes", "Edges", "Wall Clock Time (ns)", "CPU Time (ns)", "Sample Type"};
      writer.write_row(headings.begin(), headings.end());
//      cout << filename << endl;
  }
  void set_timing_run_number(int run) {this->timing_run_number = run;}
#endif
};
