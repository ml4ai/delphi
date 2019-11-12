#include "AnalysisGraph.hpp"

using namespace std;
using namespace delphi::utils;
using Eigen::VectorXd;

/*
 ============================================================================
 Private: Training by MCMC Sampling
 ============================================================================
*/

// Sample elements of the stochastic transition matrix from the
// prior distribution, based on gradable adjectives.
void AnalysisGraph::sample_initial_transition_matrix_from_prior() {
  int num_verts = this->num_vertices();

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
  for (auto& [row, col] : this->beta_dependent_cells) {
    this->A_original(row * 2, col * 2 + 1) =
        this->A_beta_factors[row][col]->compute_cell(this->graph);
  }
}

void AnalysisGraph::set_initial_latent_state_from_observed_state_sequence() {
  int num_verts = this->num_vertices();

  this->set_default_initial_state();

  for (int v = 0; v < num_verts; v++) {
    vector<Indicator>& indicators = (*this)[v].indicators;
    vector<double> next_state_values;
    for (int i = 0; i < indicators.size(); i++) {
      Indicator& ind = indicators[i];

      double ind_mean = ind.get_mean();

      while (ind_mean == 0) {
        ind_mean = this->norm_dist(this->rand_num_generator);
      }
      double next_ind_value;
      if (this->observed_state_sequence[1][v][i].empty()) {
        next_ind_value = 0;
      }
      else {
        next_ind_value =
            delphi::utils::mean(this->observed_state_sequence[1][v][i]);
      }
      next_state_values.push_back(next_ind_value / ind_mean);
    }
    double diff = delphi::utils::mean(next_state_values) - this->s0(2 * v);
    this->s0(2 * v + 1) = diff;
  }
}

void AnalysisGraph::set_initial_latent_from_end_of_training() {
  using delphi::utils::mean;
  int num_verts = this->num_vertices();

  this->set_default_initial_state();

  for (int v = 0; v < num_verts; v++) {
    vector<Indicator>& indicators = (*this)[v].indicators;
    vector<double> state_values;
    for (int i = 0; i < indicators.size(); i++) {
      Indicator& ind = indicators[i];

      double last_ind_value;
      if (this->observed_state_sequence[this->observed_state_sequence.size() -
                                        1][v][i]
              .empty()) {
        last_ind_value = 0.;
      }
      else {
        last_ind_value = mean(
            this->observed_state_sequence[this->observed_state_sequence.size() -
                                          1][v][i]);
      }
      double prev_ind_value;
      if (this->observed_state_sequence[this->observed_state_sequence.size() -
                                        2][v][i]
              .empty()) {
        prev_ind_value = 0.;
      }
      else {
        prev_ind_value = mean(
            this->observed_state_sequence[this->observed_state_sequence.size() -
                                          2][v][i]);
      }
      while (prev_ind_value == 0.) {
        prev_ind_value = this->norm_dist(this->rand_num_generator);
      }
      state_values.push_back((last_ind_value - prev_ind_value) /
                             prev_ind_value);
    }
    double diff = mean(state_values);
    this->s0(2 * v + 1) = diff;
  }
}

void AnalysisGraph::set_log_likelihood() {
  this->previous_log_likelihood = this->log_likelihood;
  this->log_likelihood = 0.0;

  for (int ts = 0; ts < this->n_timesteps; ts++) {
    this->set_current_latent_state(ts);

    // Access
    // observed_state[ vertex ][ indicator ]
    const vector<vector<vector<double>>>& observed_state =
        this->observed_state_sequence[ts];

    for (int v : this->node_indices()) {
      const int& num_inds_for_v = observed_state[v].size();

      for (int i = 0; i < observed_state[v].size(); i++) {
        const Indicator& ind = this->graph[v].indicators[i];
        for (int j = 0; j < observed_state[v][i].size(); j++) {
          const double& value = observed_state[v][i][j];
          // Even indices of latent_state keeps track of the state of each
          // vertex
          double log_likelihood_component = log_normpdf(
              value, this->current_latent_state[2 * v] * ind.mean, ind.stdev);
          this->log_likelihood += log_likelihood_component;
        }
      }
    }
  }
}

void AnalysisGraph::set_current_latent_state(int ts) {
  const Eigen::MatrixXd& A_t = tuning_param * ts * this->A_original;
  this->current_latent_state = A_t.exp() * this->s0;
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
    this->revert_back_to_previous_state();
  }
}

void AnalysisGraph::sample_from_proposal() {
  // Randomly pick an edge ≡ β
  boost::iterator_range edge_it = this->edges();

  vector<EdgeDescriptor> e(1);
  sample(
      edge_it.begin(), edge_it.end(), e.begin(), 1, this->rand_num_generator);

  // Remember the previous β
  this->previous_beta = make_pair(e[0], this->graph[e[0]].beta);

  // Perturb the β
  // TODO: Check whether this perturbation is accurate
  graph[e[0]].beta += this->norm_dist(this->rand_num_generator);

  this->update_transition_matrix_cells(e[0]);
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
    // ( 2*row, 2*col+1 ) is the transition mateix cell that got changed.
    // this->A_cells_changed.push_back( make_tuple( row, col, A( row * 2, col
    // * 2 + 1 )));

    this->A_original(row * 2, col * 2 + 1) =
        this->A_beta_factors[row][col]->compute_cell(this->graph);
  }
}

double AnalysisGraph::calculate_delta_log_prior() {
  KDE& kde = this->graph[this->previous_beta.first].kde;

  // We have to return: log( p( β_new )) - log( p( β_old ))
  return kde.logpdf(this->graph[this->previous_beta.first].beta) -
         kde.logpdf(this->previous_beta.second);
}

void AnalysisGraph::revert_back_to_previous_state() {
  this->log_likelihood = this->previous_log_likelihood;

  this->graph[this->previous_beta.first].beta = this->previous_beta.second;

  // Reset the transition matrix cells that were changed
  // TODO: Can we change the transition matrix only when the sample is
  // accepted?
  this->update_transition_matrix_cells(this->previous_beta.first);
}



/*
 ============================================================================
 Public: Training by MCMC Sampling
 ============================================================================
*/

void AnalysisGraph::set_default_initial_state() {
  // Let vertices of the CAG be v = 0, 1, 2, 3, ...
  // Then,
  //    indexes 2*v keeps track of the state of each variable v
  //    indexes 2*v+1 keeps track of the state of ∂v/∂t
  int num_verts = this->num_vertices();
  int num_els = num_verts * 2;

  this->s0 = VectorXd(num_els);
  this->s0.setZero();

  for (int i = 0; i < num_els; i += 2) {
    this->s0(i) = 1.0;
  }
}

