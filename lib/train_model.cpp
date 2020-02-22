#include "AnalysisGraph.hpp"
#include "data.hpp"
#include <tqdm.hpp>
#include <range/v3/all.hpp>

using namespace std;
using tq::trange;

void AnalysisGraph::train_model(int start_year,
                                int start_month,
                                int end_year,
                                int end_month,
                                int res,
                                int burn,
                                string country,
                                string state,
                                string county,
                                map<string, string> units,
                                InitialBeta initial_beta,
                                bool use_heuristic) {

  this->initialize_random_number_generator();
  this->uni_disc_dist = uniform_int_distribution<int>(0, this->num_nodes() - 1);

  this->construct_beta_pdfs();
  this->find_all_paths();
  this->data_heuristic = use_heuristic;

  this->n_timesteps = this->calculate_num_timesteps(
      start_year, start_month, end_year, end_month);
  this->res = res;
  this->init_betas_to(initial_beta);
  this->set_transition_matrix_from_betas();
  this->set_default_initial_state();
  this->parameterize(country, state, county, start_year, start_month, units);

  this->training_range = make_pair(make_pair(start_year, start_month),
                                   make_pair(end_year, end_month));

  if (!synthetic_data_experiment) {
    this->set_observed_state_sequence_from_data(
        start_year, start_month, end_year, end_month, country, state, county);
  }

  //this->set_initial_latent_state_from_observed_state_sequence();

  this->set_log_likelihood();

  // Accumulates the transition matrices for accepted samples
  // Access: [ sample number ]
  // training_sampled_transition_matrix_sequence.clear();
  // training_sampled_transition_matrix_sequence =
  //    vector<Eigen::MatrixXd>(this->res);
  //
  // generate_prediction()      uses
  // sample_from_likelihood. It uses
  // transition_matrix_collection
  // So to keep things simple for the moment
  // I had to fall back to
  // transition_matrix_collection
  // HOWEVER: The purpose of transition_matrix_collection
  // seem to be different in the prevous code than here.
  // In the earlier code, (in sample_from_prior()) this is
  // populated with DEFAULT_N_SAMPLES of initial transition matrices.
  // Here we populate it with res number of sampler emitted transition
  // matrices.
  //
  this->transition_matrix_collection.clear();
  this->initial_latent_state_collection.clear();
  this->transition_matrix_collection = vector<Eigen::MatrixXd>(this->res);
  this->initial_latent_state_collection = vector<Eigen::VectorXd>(this->res);

  for (int i : trange(burn)) {
    this->sample_from_posterior();
  }

  for (int i : trange(this->res)) {
    this->sample_from_posterior();
    this->transition_matrix_collection[i] = this->A_original;
    this->initial_latent_state_collection[i] = this->s0;
  }

  this->trained = true;
  RNG::release_instance();
  return;
}


/*
 ============================================================================
 Private: Get Training Data Sequence
 ============================================================================
*/

void AnalysisGraph::set_observed_state_sequence_from_data(int start_year,
                                                          int start_month,
                                                          int end_year,
                                                          int end_month,
                                                          string country,
                                                          string state,
                                                          string county) {
  this->observed_state_sequence.clear();

  // Access
  // [ timestep ][ vertex ][ indicator ]
  this->observed_state_sequence = ObservedStateSequence(this->n_timesteps);

  int year = start_year;
  int month = start_month;

  for (int ts = 0; ts < this->n_timesteps; ts++) {
    this->observed_state_sequence[ts] =
        get_observed_state_from_data(year, month, country, state, county);

    if (month == 12) {
      year++;
      month = 1;
    }
    else {
      month++;
    }
  }
}

vector<vector<vector<double>>> AnalysisGraph::get_observed_state_from_data(
    int year, int month, string country, string state, string county) {
  using ranges::to;
  using ranges::views::transform;

  int num_verts = this->num_vertices();

  // Access
  // [ vertex ][ indicator ]
  vector<vector<vector<double>>> observed_state(num_verts);

  for (int v = 0; v < num_verts; v++) {
    vector<Indicator>& indicators = (*this)[v].indicators;

    for (auto& ind : indicators) {
      auto vals = get_data_value(ind.get_name(),
                                 country,
                                 state,
                                 county,
                                 year,
                                 month,
                                 ind.get_unit(),
                                 this->data_heuristic);

      observed_state[v].push_back(vals);
    }
  }

  return observed_state;
}
