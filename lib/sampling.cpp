#include "AnalysisGraph.hpp"

using namespace std;
using namespace delphi::utils;
using delphi::utils::mean;
using Eigen::VectorXd;

/*
 ============================================================================
 Private: Training by MCMC Sampling
 ============================================================================
*/

void AnalysisGraph::set_base_transition_matrix() {
  int num_verts = this->num_vertices();

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
   *
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
  this->A_original = Eigen::MatrixXd::Zero(num_verts * 2, num_verts * 2);

  if (this->continuous) {
    for (int vert = 0; vert < 2 * num_verts; vert += 2) {
        this->A_original(vert, vert + 1) = 1;
    }
  }
  else {
    // Discretized version
    // A_d = I + A_c × Δt
    // Fill the Δts
    //for (int vert = 0; vert < 2 * num_verts; vert += 2) {
    //    this->A_original(vert, vert + 1) = this->delta_t;
    //}
    for (int vert = 0; vert < 2 * num_verts; vert++) {
        // Filling the diagonal (Adding I)
        this->A_original(vert, vert) = 1;

        if (vert % 2 == 0) {
            // Fill the Δts
            this->A_original(vert, vert + 1) = this->delta_t;
        }
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

  if (this->continuous) {
    // Initialize the transition matrix pre-calculation data structure.
    // This data structure holds all the transition matrices required to
    // advance the system from each timestep to the next.
    for (double gap : set<double>(this->observation_timestep_gaps.begin(),
                                  this->observation_timestep_gaps.end())) {
      this->e_A_ts.insert(make_pair(gap, (this->A_original * gap).exp()));
    }
  }
}

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
        // Precalculate all the transition matrices required to advance the system
        // from one timestep to the next.
        for (auto [gap, mat] : this->e_A_ts) {
          this->e_A_ts[gap] = (this->A_original * gap).exp();
        }
      }
      this->current_latent_state = this->s0;

//      this->update_independent_node_latent_state_with_generated_derivatives(
//          0, this->generated_concept, this->generated_latent_sequence);
      this->update_latent_state_with_generated_derivatives(0);

      this->set_log_likelihood_helper(0);

      for (int ts = 1; ts < this->n_timesteps; ts++) {
        this->current_latent_state =
            this->e_A_ts[this->observation_timestep_gaps[ts]]
            * this->current_latent_state;
//        this->update_independent_node_latent_state_with_generated_derivatives(
//            ts, this->generated_concept, this->generated_latent_sequence);
        this->update_latent_state_with_generated_derivatives(ts);
        this->set_log_likelihood_helper(ts);
      }
  } else {
      // Discretized version
      this->current_latent_state = this->s0;

      for (int ts = 0; ts < this->n_timesteps; ts++) {

          // Set derivatives for frozen nodes
          for (const auto & [ v, deriv_func ] : this->external_concepts) {
              const Indicator& ind = this->graph[v].indicators[0];
              this->current_latent_state[2 * v + 1] = deriv_func(ts, ind.mean);
          }

//          this->update_independent_node_latent_state_with_generated_derivatives(
//              ts, this->generated_concept, this->generated_latent_sequence);
          this->update_latent_state_with_generated_derivatives(ts);
          set_log_likelihood_helper(ts);
          this->current_latent_state = this->A_original * this->current_latent_state;
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
//    else {
//        Node& n = (*this)[this->generated_concept];
//        n.mean = delphi::utils::mean(this->generated_latent_sequence);
//        n.std = delphi::utils::standard_deviation(
//            n.mean, this->generated_latent_sequence);
//        this->partition_data_and_calculate_mean_std_for_each_partition
//            (n, this->generated_latent_sequence);
//    }
  }
//  else {
//    if (this->generated_concept > -1) {
//      Node& n = (*this)[this->generated_concept];
//      n.mean = delphi::utils::mean(this->generated_latent_sequence);
//      n.std = delphi::utils::standard_deviation(
//          n.mean, this->generated_latent_sequence);
//      this->partition_data_and_calculate_mean_std_for_each_partition
//                                     (n, this->generated_latent_sequence);
//    }
//  }
}

void AnalysisGraph::sample_from_proposal() {
  // Flip a coin and decide whether to perturb a θ or a derivative
  this->coin_flip = this->uni_dist(this->rand_num_generator);
  this->generated_concept = -1;

  if (this->coin_flip < this->coin_flip_thresh) {
    // Randomly pick an edge ≡ θ
    boost::iterator_range edge_it = this->edges();

    vector<EdgeDescriptor> e(1);
    sample(
        edge_it.begin(), edge_it.end(), e.begin(), 1, this->rand_num_generator);

    // Remember the previous θ
    this->previous_theta = make_pair(e[0], this->graph[e[0]].theta);

    // Perturb the θ
    // TODO: Check whether this perturbation is accurate
    this->graph[e[0]].theta += this->norm_dist(this->rand_num_generator);

    this->update_transition_matrix_cells(e[0]);
  }
  else {
    // Randomly select a concept
    int concept = this->concept_sample_pool[this->uni_disc_dist(this->rand_num_generator)];
    this->changed_derivative = 2 * concept + 1;

    if (this->independent_nodes.find(concept) != this->independent_nodes.end()) {
      this->generated_concept = concept;
      Node &n = (*this)[this->generated_concept];
      this->generate_from_data_mean_and_std_gussian(
          n.mean, n.std, this->n_timesteps,
          n.partition_mean_std, n.period);
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
    KDE& kde = this->graph[this->previous_theta.first].kde;

    // We have to return: log( p( θ_new )) - log( p( θ_old ))
    return kde.logpdf(this->graph[this->previous_theta.first].theta) -
           kde.logpdf(this->previous_theta.second);
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
      return (1 - this->generated_latent_sequence[0]) *
             (1 + this->generated_latent_sequence[0]) /
             (2 * 0.01);
    }
  }
}

void AnalysisGraph::revert_back_to_previous_state() {
  this->log_likelihood = this->previous_log_likelihood;

  if (this->coin_flip < this->coin_flip_thresh) {
    // A θ has been sampled
    this->graph[this->previous_theta.first].theta = this->previous_theta.second;

    // Reset the transition matrix cells that were changed
    // TODO: Can we change the transition matrix only when the sample is
    // accepted?
    this->update_transition_matrix_cells(this->previous_theta.first);
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
  int num_els = this->num_vertices() * 2;

  this->s0 = VectorXd(num_els);
  this->s0.setZero();

  for (int i = 0; i < num_els; i += 2) {
    this->s0(i) = 1.0;
  }

  if (id == InitialDerivative::DERI_PRIOR) {
    double derivative_prior_std = sqrt(this->derivative_prior_variance);
    for (int i = 1; i < num_els; i += 2) {
      this->s0(i) = derivative_prior_std *
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
