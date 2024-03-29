#include "AnalysisGraph.hpp"
#ifdef _OPENMP
    #include <omp.h>
#endif

#ifdef TIME
  #include "Timer.hpp"
#endif

using namespace std;
using namespace delphi::utils;
using delphi::utils::mean;
using Eigen::VectorXd;

/*
 ============================================================================
 Private: Training by MCMC Sampling
 ============================================================================
*/

void AnalysisGraph::assemble_base_LDS(InitialDerivative id) {
    if (this->continuous) {
        // Initialize matrix exponential pre-calculation related data structures.
        unordered_set<double> gaps_set;

        if (this->head_node_model == HeadNodeModel::HNM_NAIVE) {
            gaps_set.insert(1); // Due to the current head node model
        } else { /// HeadNodeModel::HNM_FOURIER
            gaps_set = unordered_set<double>(this->modeling_timestep_gaps.begin() + 1,
                                             this->modeling_timestep_gaps.end());
        }
        this->observation_timestep_unique_gaps = vector<double>(gaps_set.begin(),
                                                                gaps_set.end());
        #ifdef MULTI_THREADING
            this->matrix_exponential_futures = vector<future<Eigen::MatrixXd>>(
                this->observation_timestep_unique_gaps.size());
        #endif
    }

    this->derivative_prior_variance = 0.1;
    this->set_default_initial_state(id);

    #ifdef MULTI_THREADING
        vector<future<Eigen::MatrixXd>> fourier_me_futures(
                                 this->observation_timestep_unique_gaps.size());
    #endif

    if (this->head_node_model == HeadNodeModel::HNM_FOURIER) {
        auto [A_fourier_full_base, s0_fourier_full] =
                       fit_seasonal_head_node_model_via_fourier_decomposition();
        this->A_fourier_base = A_fourier_full_base;
        this->s0_fourier = s0_fourier_full;

        if (this->continuous) {
            #ifdef MULTI_THREADING
                this->compute_multiple_matrix_exponentials_parallelly(
                                      this->A_fourier_base, fourier_me_futures);
            #else
                for (double gap : this->observation_timestep_unique_gaps) {
                    this->e_A_fourier_ts[gap] =
                                             (this->A_fourier_base * gap).exp();
                }
            #endif
        } else { /// Discrete
            this->A_fourier_base = (this->A_fourier_base * this->delta_t).exp();
        }

        this->current_latent_state = this->s0_fourier;
        this->current_latent_state.head(this->s0.size()) += this->s0;
    }

    this->set_transition_matrix_from_betas();

    #ifdef MULTI_THREADING
        if (this->continuous) {
            this->compute_multiple_matrix_exponentials_parallelly(
                                              this->A_original,
                                              this->matrix_exponential_futures);

            if (this->head_node_model == HeadNodeModel::HNM_FOURIER) {
                // Accumulate the precalculated exponentiated transition
                // matrices from different threads
                for (int i = 0;
                     i < this->observation_timestep_unique_gaps.size(); i++) {
                    double &gap = this->observation_timestep_unique_gaps[i];
                    this->e_A_fourier_ts[gap] = fourier_me_futures[i].get();
                }
            }
        }
    #endif
}

void AnalysisGraph::set_base_transition_matrix() {
  int n_verts = this->num_vertices();
  int lds_size = 2 * n_verts;

  // A base transition matrix with the entries that does not change across
  // samples (A_c : continuous).
  /*
   *          0  1  2  3  4  5
   *  var_1 | 0  1  0     0    | 0
   *        | 0  0  0  0  0  0 | 1 ∂var_1 / ∂t
   *  var_2 | 0     0  1  0    | 2
   *        | 0  0  0  0  0  0 | 3
   *  var_3 | 0     0     0  1 | 4
   *        | 0  0  0  0  0  0 | 5
   */

  // A base transition matrix with the entries that does not change across
  // samples (A_d : discretized).
  /*
   *          0  1  2  3  4  5
   *  var_1 | 1 Δt  0     0    | 0
   *        | 0  1  0  0  0  0 | 1 ∂var_1 / ∂t
   *  var_2 | 0     1 Δt  0    | 2
   *        | 0  0  0  1  0  0 | 3
   *  var_3 | 0     0     1 Δt | 4
   *        | 0  0  0  0  0  1 | 5
   *
   *  Based on the directed simple paths in the CAG, some of the remaining
   *  blank cells would be filled with β related values
   *  If we include second order derivatives to the model, there would be
   *  three rows for each variable and some of the off diagonal elements
   *  of rows with index % 3 = 1 would be non zero.
   */
  this->A_original = Eigen::MatrixXd::Zero(lds_size, lds_size);

  if (this->continuous) {
    for (int vert = 0; vert < n_verts; vert++) {
        if (this->head_node_model == HeadNodeModel::HNM_FOURIER &&
                          this->head_nodes.find(vert) != this->head_nodes.end())
            continue;
        int dot_row = 2 * vert;
        this->A_original(dot_row, dot_row + 1) = 1;
    }
  }
  else {
    // Discretized version
    // A_d = I + A_c × Δt
    // Fill the Δts
    for (int vert = 0; vert < n_verts; vert++) {
        int dot_row = 2 * vert;
        int dot_dot_row = dot_row + 1;

        // Filling the diagonal (Adding I)
        this->A_original(dot_row, dot_row) = 1;
        this->A_original(dot_dot_row, dot_dot_row) = 1;

        if (this->head_node_model == HeadNodeModel::HNM_FOURIER &&
            this->head_nodes.find(vert) != this->head_nodes.end())
            continue;

        // Fill the Δts
        this->A_original(dot_row, dot_dot_row) = this->delta_t;
    }
  }
}

// Initialize elements of the stochastic transition matrix from the
// prior distribution, based on gradable adjectives.
void AnalysisGraph::set_transition_matrix_from_betas() {

  this->set_base_transition_matrix();

  // Update the β factor dependent cells of this matrix
  for (auto& [row, col] : this->beta_dependent_cells) {
    this->A_original(row * 2, col * 2 + 1) =
        this->A_beta_factors[row][col]->compute_cell(this->graph);
  }
}


#ifdef MULTI_THREADING
    static Eigen::MatrixXd compute_matrix_exponential(
                        const Eigen::Ref<const Eigen::MatrixXd> A, double gap) {
        return (A * gap).exp();
    }


    void AnalysisGraph::compute_multiple_matrix_exponentials_parallelly(
                                  const Eigen::MatrixXd & A,
                                  vector<future<Eigen::MatrixXd>> &me_futures) {
        // Create threads to precalculate all the transition matrices required to
        // advance the system from one time step to the next by computing matrix
        // exponentials in parallel - e^(this->A_original * gap)
        for (int i = 0; i < this->observation_timestep_unique_gaps.size(); i++) {
            double &gap = this->observation_timestep_unique_gaps[i];
            me_futures[i] = async(launch::async,
                                  compute_matrix_exponential, A, gap);
        }
    }
#endif

void AnalysisGraph::set_log_likelihood_helper(int ts) {
    // Access (concept is a vertex in the CAG)
    // observed_state[ concept ][ indicator ][ observation ]
    const vector<vector<vector<double>>>& observed_state =
        this->observed_state_sequence[ts];

    for (int v : this->node_indices()) {
        const int& num_inds_for_v = observed_state[v].size();

        for (int i = 0; i < observed_state[v].size(); i++) {
            const Indicator& ind = this->graph[v].indicators[i];
            for (int o = 0; o < observed_state[v][i].size(); o++) {
                const double& obs = observed_state[v][i][o];
                // Even indices of latent_state keeps track of the state of each
                // vertex
                double log_likelihood_component = log_normpdf(
                        obs, this->current_latent_state[2 * v] * ind.mean, ind.stdev);
                this->log_likelihood += log_likelihood_component;
            }
        }
    }
}

void AnalysisGraph::set_log_likelihood() {
  this->previous_log_likelihood = this->log_likelihood;
  this->log_likelihood = 0.0;

  if (this->observed_state_sequence.empty()) {
    return;
  }

  if (this->continuous) {
      if (this->coin_flip < this->coin_flip_thresh) {
        // A θ has been sampled
        {
          #ifdef TIME
              this->mcmc_part_duration.second.clear();
              this->mcmc_part_duration.second.push_back(this->timing_run_number);
              this->mcmc_part_duration.second.push_back(this->num_nodes());
              this->mcmc_part_duration.second.push_back(this->num_edges());
              Timer t_part = Timer("ME", this->mcmc_part_duration);
          #endif

          #ifdef MULTI_THREADING
              // Accumulate the precalculated exponentiated transition matrices
              // from different threads
              for (int i = 0; i < this->observation_timestep_unique_gaps.size();
                   i++) {
                  double &gap = this->observation_timestep_unique_gaps[i];
                  this->e_A_ts[gap] = this->matrix_exponential_futures[i].get();
              }
          /*
          #ifdef _OPENMP
              this->e_A_ts.clear();
              #pragma omp parallel
              {
                  unordered_map<double, Eigen::MatrixXd> partial_e_A_ts;
                  for (int i = 0; i < this->observation_timestep_unique_gaps.size();
                       i++) {
                      int gap = this->observation_timestep_unique_gaps[i];
                      partial_e_A_ts[gap] = (this->A_original * gap).exp();
                  }
                  #pragma omp critical
                  this->e_A_ts.merge(partial_e_A_ts);
                  #pragma omp barrier
              }
          */
          #else
              for (double gap : this->observation_timestep_unique_gaps) {
                this->e_A_ts[gap] = (this->A_original * gap).exp();
              }
          #endif

          if (this->head_node_model == HeadNodeModel::HNM_FOURIER) {
            // Merge exponentiated Fourier decomposition based head node model
            // transition matrices with the exponentiated transition matrices
            // defining the relationships between concepts.
            for (double gap : this->observation_timestep_unique_gaps) {
                this->e_A_fourier_ts.at(gap).topLeftCorner(
                            this->e_A_ts.at(gap).rows(),
                            this->e_A_ts.at(gap).cols()) = this->e_A_ts.at(gap);
                this->e_A_ts[gap] = this->e_A_fourier_ts.at(gap);
            }
          }
        }
        #ifdef TIME
          this->mcmc_part_duration.second.push_back(11);
          this->writer.write_row(this->mcmc_part_duration.second.begin(),
                                 this->mcmc_part_duration.second.end());
        #endif
      }

      {
        #ifdef TIME
          this->mcmc_part_duration.second.clear();
          this->mcmc_part_duration.second.push_back(this->timing_run_number);
          this->mcmc_part_duration.second.push_back(this->num_nodes());
          this->mcmc_part_duration.second.push_back(this->num_edges());
          Timer t_part = Timer("Log Likelihood", this->mcmc_part_duration);
        #endif
        if (this->head_node_model == HeadNodeModel::HNM_NAIVE) {
          this->current_latent_state = this->s0;
          int ts_monthly = 0;
          for (int ts = 0; ts < this->n_timesteps; ts++) {

            // Set derivatives for frozen nodes
            for (const auto& [v, deriv_func] : this->external_concepts) {
              const Indicator& ind = this->graph[v].indicators[0];
              this->current_latent_state[2 * v + 1] = deriv_func(ts, ind.mean);
            }

            // For the current naive seasonal head node model to work, we have
            // to advance the system on modeling timestep at a time for all the
            // timesteps where there are no observations till we hit a
            // timestep with data to compute the log likelihood
            for (int ts_gap = 0; ts_gap < this->modeling_timestep_gaps[ts];
                                                                     ts_gap++) {
              this->update_latent_state_with_generated_derivatives(ts_monthly,
                                                                ts_monthly + 1);
              this->current_latent_state =
                               this->e_A_ts.at(1) * this->current_latent_state;
              ts_monthly++;
            }
            this->set_log_likelihood_helper(ts);
          }
        } else { /// HeadNodeModel::HNM_FOURIER
          // Merge the initial state for concepts with the initial state for
          // Fourier decomposition based seasonal head node model.
          this->current_latent_state = this->s0_fourier;
          this->current_latent_state.head(this->s0.size()) += this->s0;

          this->set_log_likelihood_helper(0);

          for (int ts = 1; ts < this->n_timesteps; ts++) {
            this->current_latent_state =
                    this->e_A_ts.at(this->modeling_timestep_gaps[ts])
                                                   * this->current_latent_state;
            this->set_log_likelihood_helper(ts);
          }
        }
      }
      #ifdef TIME
        this->mcmc_part_duration.second.push_back(13);
        this->writer.write_row(this->mcmc_part_duration.second.begin(),
                               this->mcmc_part_duration.second.end());
      #endif
  } else {
      // Discretized version
      if (this->head_node_model == HeadNodeModel::HNM_NAIVE) {
          this->current_latent_state = this->s0;
          int ts_monthly = 0;

          for (int ts = 0; ts < this->n_timesteps; ts++) {

              // Set derivatives for frozen nodes
              for (const auto& [v, deriv_func] : this->external_concepts) {
                  const Indicator& ind = this->graph[v].indicators[0];
                  this->current_latent_state[2 * v + 1] =
                                                       deriv_func(ts, ind.mean);
              }

              for (int ts_gap = 0; ts_gap < this->modeling_timestep_gaps[ts];
                                                                     ts_gap++) {
                  this->update_latent_state_with_generated_derivatives(
                                                    ts_monthly, ts_monthly + 1);
                  this->current_latent_state =
                                  this->A_original * this->current_latent_state;
                  ts_monthly++;
              }
              this->set_log_likelihood_helper(ts);
          }
      } else { /// HeadNodeModel::HNM_FOURIER
          this->merge_concept_LDS_into_seasonal_head_node_modeling_LDS(
                                                    this->A_original, this->s0);

          for (int ts = 0; ts < this->n_timesteps; ts++) {
              this->set_log_likelihood_helper(ts);
              this->current_latent_state = this->A_fourier_base *
                                                     this->current_latent_state;
          }
      }
  }
}

void AnalysisGraph::sample_from_posterior() {
  // Sample a new transition matrix from the proposal distribution
  this->sample_from_proposal();

  double delta_log_prior = this->calculate_delta_log_prior();

  this->set_log_likelihood();
  double delta_log_likelihood =
      this->log_likelihood - this->previous_log_likelihood;

  double delta_log_joint_probability = delta_log_prior + delta_log_likelihood;

  double acceptance_probability = min(1.0, exp(delta_log_joint_probability));

  if (acceptance_probability < this->uni_dist(this->rand_num_generator)) {
    // Reject the sample
    if (this->generated_concept == -1) {
      this->revert_back_to_previous_state();
    }
    this->log_likelihood = this->previous_log_likelihood;
  }
//  else {
//    if (this->generated_concept > -1) {
//      Node& n = (*this)[this->generated_concept];
//      this->partition_data_and_calculate_mean_std_for_each_partition
//                                     (n, this->generated_latent_sequence);
//    }
//  }
}

void AnalysisGraph::sample_from_proposal() {
  // Flip a coin and decide whether to perturb a θ or a derivative
  if (this->edge_sample_pool.empty()) {
      // All edge weights are frozen. Always sample a concept.
      this->coin_flip = this->coin_flip_thresh;
  } else {
      this->coin_flip = this->uni_dist(this->rand_num_generator);
  }
  this->generated_concept = -1;

  if (this->coin_flip < this->coin_flip_thresh) {
    // Randomly pick an edge ≡ θ
    EdgeDescriptor ed = this->edge_sample_pool[this->uni_disc_dist_edge(this->rand_num_generator)];

    // Remember the previous θ and logpdf(θ)
    this->previous_theta = make_tuple(ed, this->graph[ed].get_theta(), this->graph[ed].logpdf_theta);

    // Remember the previously calculated A_original for the previous edge
    // weight before we update it to match the newly sampled edge. This way, if
    // this sample gets rejected, we do not have to re-compete A_original, that
    // involve computing the chain rule, which is an expensive operation for
    // larger CAGs.
    this->previous_A_original = this->A_original;

    // Remember the previously calculated matrix exponentials for the previous
    // transition matrix before we update e_A_ts with the matrix exponentials
    // for the newly sampled transition matrix. This way, if this sample gets
    // rejected, we do not have to re-compete the matrix exponentials for the
    // previous transition matrix, which is as expensive operation.
    this->previous_e_A_ts = this->e_A_ts;


    // Perturb the θ and compute the new logpdf(θ)
    this->graph[ed].set_theta(this->graph[ed].get_theta() + this->norm_dist(this->rand_num_generator) / 10);
    {
      #ifdef TIME
        this->mcmc_part_duration.second.clear();
        this->mcmc_part_duration.second.push_back(this->timing_run_number);
        this->mcmc_part_duration.second.push_back(this->num_nodes());
        this->mcmc_part_duration.second.push_back(this->num_edges());
        Timer t_part = Timer("KDE", this->mcmc_part_duration);
      #endif
      this->graph[ed].compute_logpdf_theta();
    }
    #ifdef TIME
      this->mcmc_part_duration.second.push_back(10);
      this->writer.write_row(this->mcmc_part_duration.second.begin(),
                             this->mcmc_part_duration.second.end());
    #endif

    {
      #ifdef TIME
        this->mcmc_part_duration.second.clear();
        this->mcmc_part_duration.second.push_back(this->timing_run_number);
        this->mcmc_part_duration.second.push_back(this->num_nodes());
        this->mcmc_part_duration.second.push_back(this->num_edges());
        Timer t_part = Timer("UPTM", this->mcmc_part_duration);
      #endif
      this->update_transition_matrix_cells(ed);
    }

    #ifdef MULTI_THREADING
        this->compute_multiple_matrix_exponentials_parallelly(this->A_original,
                                              this->matrix_exponential_futures);
    #endif

    #ifdef TIME
      this->mcmc_part_duration.second.push_back(12);
      this->writer.write_row(this->mcmc_part_duration.second.begin(),
                             this->mcmc_part_duration.second.end());
    #endif
  }
  else {
    // Randomly select a concept
    int concept = this->concept_sample_pool[this->uni_disc_dist(this->rand_num_generator)];
    this->changed_derivative = 2 * concept + 1;

    if (this->head_nodes.find(concept) != this->head_nodes.end()) {
        if (this->head_node_model == HeadNodeModel::HNM_NAIVE) {
            this->generated_concept = concept;
            this->generate_head_node_latent_sequence(
                this->generated_concept, this->n_timesteps, true, 0);
        } else {
            /// HeadNodeModel::HNM_FOURIER
            // TODO: What should we do?
        }
    }
    else {
      // to change the derivative
      this->previous_derivative = this->s0[this->changed_derivative];
      this->s0[this->changed_derivative] +=
          this->norm_dist(this->rand_num_generator);
    }
  }
}

void AnalysisGraph::update_transition_matrix_cells(EdgeDescriptor e) {
  pair<int, int> beta =
      make_pair(boost::source(e, this->graph), boost::target(e, this->graph));

  pair<MMapIterator, MMapIterator> beta_dept_cells =
      this->beta2cell.equal_range(beta);

  // TODO: I am introducing this to implement calculate_Δ_log_prior
  // Remember the cells of A that got changed and their previous values
  // this->A_cells_changed.clear();

  for (MMapIterator it = beta_dept_cells.first; it != beta_dept_cells.second;
       it++) {
    int& row = it->second.first;
    int& col = it->second.second;

    // Note that I am remembering row and col instead of 2*row and 2*col+1
    // row and col resembles an edge in the CAG: row -> col
    // ( 2*row, 2*col+1 ) is the transition matrix cell that got changed.
    // this->A_cells_changed.push_back( make_tuple( row, col, A( row * 2, col
    // * 2 + 1 )));

    this->A_original(row * 2, col * 2 + 1) =
        this->A_beta_factors[row][col]->compute_cell(this->graph);
  }
}

double AnalysisGraph::calculate_delta_log_prior() {
  if (this->coin_flip < this->coin_flip_thresh) {
    // A θ has been sampled
    // KDE& kde = this->graph[get<0>(this->previous_theta)].kde;

    // We have to return: log( p( θ_new )) - log( p( θ_old ))
    //    return kde.logpdf(this->graph[this->previous_theta.first].theta) -
    //           kde.logpdf(this->previous_theta.second);
    return this->graph[get<0>(this->previous_theta)].logpdf_theta -
           get<2>(this->previous_theta);
  }
  else {
    if (this->generated_concept == -1) {
      // A derivative  has been sampled
      // We assume the prior for derivative is N(0, 0.1)
      // We have to return: log( p(ẋ_new )) - log( p( ẋ_old ))
      // After some mathematical simplifications we can derive
      // (ẋ_old - ẋ_new)(ẋ_old + ẋ_new) / 2σ²
      return (this->previous_derivative - this->s0[this->changed_derivative]) *
             (this->previous_derivative + this->s0[this->changed_derivative]) /
             (2 * this->derivative_prior_variance);
    }
    else {
      // A derivative sequence for an independent node has been generated
      // When latent state at ts = 0 is 1, it makes the observation 0 the
      // highest probable value.
      // The standard deviation of ts = 0 latent state is set to 0.01
      /*
      return (1 - this->generated_latent_sequence[0]) *
             (1 + this->generated_latent_sequence[0]) /
             (2 * 0.01);
      */
      return 0;
    }
  }
}

void AnalysisGraph::revert_back_to_previous_state() {
  this->log_likelihood = this->previous_log_likelihood;

  if (this->coin_flip < this->coin_flip_thresh) {
    // A θ has been sampled
    EdgeDescriptor perturbed_edge = get<0>(this->previous_theta);

    this->graph[perturbed_edge].set_theta(get<1>(this->previous_theta));
    this->graph[perturbed_edge].logpdf_theta = get<2>(this->previous_theta);

    // Reset the transition matrix cells that were changed
    // TODO: Can we change the transition matrix only when the sample is
    // accepted?
    this->A_original = this->previous_A_original;
    this->e_A_ts = this->previous_e_A_ts;
  }
  else {
    // A derivative  has been sampled
    // this->s0 = this->s0_prev;
    s0[this->changed_derivative] = this->previous_derivative;
  }
}

/*
 ============================================================================
 Public: Training by MCMC Sampling
 ============================================================================
*/

void AnalysisGraph::set_default_initial_state(InitialDerivative id) {
  // Let vertices of the CAG be v = 0, 1, 2, 3, ...
  // Then,
  //    indexes 2*v keeps track of the state of each variable v
  //    indexes 2*v+1 keeps track of the state of ∂v/∂t
  int n_verts = this->num_vertices();

  this->s0 = VectorXd::Zero(2 * n_verts);

  for (int v = 0; v < n_verts; v++) {
    if (this->head_node_model == HeadNodeModel::HNM_FOURIER &&
                           this->head_nodes.find(v) != this->head_nodes.end()) {
        continue;
    }

    int value_row = 2 * v;
    this->s0(value_row) = 1.0;

    if (this->MAP_sample_number > -1) {
        // Warm start using the MAP estimate of the previous training run
        this->s0(value_row + 1) = this->initial_latent_state_collection
                                 [this->MAP_sample_number](value_row + 1);
    }
    else if (id == InitialDerivative::DERI_PRIOR) {
        double derivative_prior_std = sqrt(this->derivative_prior_variance);
        this->s0(value_row + 1) = derivative_prior_std *
                                   this->norm_dist(this->rand_num_generator);
    }
  }
}

void AnalysisGraph::set_res(size_t res) {
    this->res = res;
}

size_t AnalysisGraph::get_res() {
    return this->res;
}

void AnalysisGraph::check_multithreading() {
    #ifdef _OPENMP
        std::cout << "Compiled with OpenMP\n";
        std::cout << "Maximum number of threads: " << omp_get_max_threads()
                  << endl;
        #pragma omp parallel
            {
                int n_threads = omp_get_num_threads();
                int tid = omp_get_thread_num();
                if (tid == 0) {
                    printf("%d Threads created\n", n_threads);
                }
                printf("Thread - %d\n", tid);
            }
    #else
        std::cout << "Compiled **without** OpenMP\n";
    #endif

    #ifdef MULTI_THREADING
        std::cout << "Computing matrix exponential for different gaps in parallel\n";
        cout << "\nMaximum number of hardware threads run parallelly: " << thread::hardware_concurrency() << endl;
    #else
        std::cout << "Computing matrix exponential for different gaps sequentially\n";
    #endif
}
