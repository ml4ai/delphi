// Modeling independent CAG nodes

#include "AnalysisGraph.hpp"

using namespace std;

void AnalysisGraph::generate_from_data_mean_and_std_gussian(double mean,
                                                            double std,
                                                            int num_timesteps) {
  this->generated_latent_sequence = vector<double>(num_timesteps);

  for (int ts = 0; ts < num_timesteps; ts++) {
    this->generated_latent_sequence[ts] = mean + std
                                      * norm_dist(this->rand_num_generator);
  }
}

void AnalysisGraph::generate_independent_node_latent_sequences(int samp, int num_timesteps) {
  for (int v : this->independent_nodes) {
    Node &n = (*this)[v];

    double mean;
    double std;

    if (samp > -1) {
      mean = this->latent_mean_collection[samp][v];
      std = this->latent_std_collection[samp][v];
    }
    else {
      mean = n.mean;
      std = n.std;
    }

    this->generate_from_data_mean_and_std_gussian(mean,
                                                  std,
                                                  num_timesteps);
    n.generated_latent_sequence.clear();
    n.generated_latent_sequence = this->generated_latent_sequence;
  }
}

void AnalysisGraph::
    update_independent_node_latent_state_with_generated_derivatives(
        int ts, int concept_id, vector<double>& latent_sequence) {
  if (concept_id > -1 && ts < latent_sequence.size() - 1) {
    this->current_latent_state[2 * concept_id + 1] = latent_sequence[ts + 1]
                                                     - latent_sequence[ts];

    if (ts == 0) {
      this->current_latent_state[2 * concept_id] = latent_sequence[0];
    }
  }
}

void AnalysisGraph::update_latent_state_with_generated_derivatives(int ts) {
  for (int v : this->independent_nodes) {
    Node &n = (*this)[v];
    this->update_independent_node_latent_state_with_generated_derivatives(ts, v, n.generated_latent_sequence);
  }
}
