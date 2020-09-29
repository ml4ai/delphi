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
                                bool use_heuristic,
                                bool use_continuous) {

  if (!synthetic_data_experiment && !causemos_call) {
    // Delphi is run locally using observation data from delphi.db
    // For a synthetic data experiment, the observed state sequence is
    // generated.
    // For a CauseMos call, the observation sequences are provided in the create
    // model JSON call and the observed state sequence is set in the method
    // AnalysisGraph::set_observed_state_sequence_from_json_data(), which is
    // defined in causemos_integration.cpp
    this->set_observed_state_sequence_from_data(start_year, start_month,
                                                    country, state, county);
  }

  //this->initialize_parameters(res, initial_beta, use_continuous);
  this->initialize_parameters(start_year, start_month, end_year, end_month,
                            res, initial_beta, use_heuristic, use_continuous);

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
                                                          string country,
                                                          string state,
                                                          string county) {
  this->observed_state_sequence.clear();

  // Access (concept is a vertex in the CAG)
  // [ timestep ][ concept ][ indicator ][ observation ]
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

  // Access (concept is a vertex in the CAG)
  // [ concept ][ indicator ][ observation ]
  vector<vector<vector<double>>> observed_state(num_verts);

  for (int v = 0; v < num_verts; v++) {
    vector<Indicator>& indicators = (*this)[v].indicators;

    for (auto& ind : indicators) {
      vector<double> vals = get_observations_for(ind.get_name(),
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
