// Modeling independent CAG nodes

#include "AnalysisGraph.hpp"
#include "dbg.h"

using namespace std;

void AnalysisGraph::partition_data_and_calculate_mean_std_for_each_partition(
    Node& n, vector<double>& latent_sequence) {
  unordered_map<int, std::vector<double>> partitioned_data;

  for (int ts = 0; ts < latent_sequence.size(); ts++) {
    int partition = ts % n.period;
    partitioned_data[partition].push_back(latent_sequence[ts]);
  }

  for (const auto & [ partition, data ] : partitioned_data) {
//    double partition_mean = delphi::utils::mean(data);
    double partition_mean = delphi::utils::median(data);
    double partition_std = 1;

    if (data.size() > 1) {
      partition_std = delphi::utils::standard_deviation(partition_mean, data);
    }

//    n.partition_mean_std[partition] = make_pair(partition_mean, partition_std);
    n.partition_mean_std[partition] = make_pair(partition_mean, 1);
  }
}


void AnalysisGraph::apply_constraint_at(int ts, int node_id) {
  if (delphi::utils::in(this->head_node_one_off_constraints, ts)) {
    vector<pair<int, double>> constraint_vec = this->head_node_one_off_constraints.at(ts);
    for (auto node_const : constraint_vec) {
      if (node_id == node_const.first) {
        this->generated_latent_sequence[ts] = node_const.second;
      }
    }
  }
}


void AnalysisGraph::generate_head_node_latent_sequence(int node_id,
                                                       int num_timesteps,
                                                       bool sample,
                                                       int seq_no) {
    Node &n = (*this)[node_id];
    this->generated_latent_sequence = vector<double>(num_timesteps, 0);

    if (!n.centers.empty()) {
      if (n.model.compare("center") == 0) {
        for (int ts = 0; ts < num_timesteps; ts++) {
          int partition = ts % n.period;
          this->generated_latent_sequence[ts] = n.centers[partition];

          apply_constraint_at(ts, node_id);
        }
      }
      else if (n.model.compare("absolute_change") == 0) {
        //        dbg("absolute_change");
        this->generated_latent_sequence[0] = n.centers[0];
        for (int ts = 0; ts < num_timesteps - 1; ts++) {
          int partition = ts % n.period;
          this->generated_latent_sequence[ts + 1] =
              this->generated_latent_sequence[ts] + n.changes[partition + 1];

          apply_constraint_at(ts + 1, node_id);
        }
      }
      else if (n.model.compare("relative_change") == 0) {
        //        dbg("relative_change");
        this->generated_latent_sequence[0] = n.centers[0];
        for (int ts = 0; ts < num_timesteps - 1; ts++) {
          int partition = ts % n.period;
          this->generated_latent_sequence[ts + 1] =
              this->generated_latent_sequence[ts] +
              n.changes[partition + 1] *
                  (this->generated_latent_sequence[ts] + 1);

          apply_constraint_at(ts + 1, node_id);
        }
      }

      if (sample) {
        for (int ts = 0; ts < num_timesteps; ts++) {
          int partition = ts % n.period;
          this->generated_latent_sequence[ts] +=
              n.spreads[partition] * norm_dist(this->rand_num_generator);
        }
      }
      else {
        int sections = 5; // an odd number
        int half_sections = (sections - 1) / 2;
        int turn = seq_no % sections;
        for (int ts = 0; ts < num_timesteps; ts++) {
          int partition = ts % n.period;
          this->generated_latent_sequence[ts] +=
              (turn - half_sections) * n.spreads[partition];
          apply_constraint_at(ts, node_id);
        }
      }

      if (n.has_max) {
        for (int ts = 0; ts < num_timesteps; ts++) {
          if (this->generated_latent_sequence[ts] > n.max_val) {
            this->generated_latent_sequence[ts] = n.max_val;
          }
        }
      }
      if (n.has_min) {
        for (int ts = 0; ts < num_timesteps; ts++) {
          if (this->generated_latent_sequence[ts] < n.min_val) {
            this->generated_latent_sequence[ts] = n.min_val;
          }
        }
      }
    }
}


void AnalysisGraph::generate_head_node_latent_sequence_from_changes(Node &n,
                                                                    int num_timesteps,
                                                                    bool sample) {
    this->generated_latent_sequence = vector<double>(num_timesteps);
    this->generated_latent_sequence[0] = n.centers[0];

    for (int ts = 0; ts < num_timesteps - 1; ts++) {
        int partition = ts % n.period;

        if (n.model.compare("absolute_change") == 0) {
            this->generated_latent_sequence[ts + 1] =
                this->generated_latent_sequence[ts] + n.changes[partition + 1];
        } else if (n.model.compare("relative_change") == 0) {
            this->generated_latent_sequence[ts + 1] =
                this->generated_latent_sequence[ts] + n.changes[partition + 1]
                                                        * (this->generated_latent_sequence[ts] + 1);
        }
    }

    if (sample) {
        for (int ts = 0; ts < num_timesteps; ts++) {
          int partition = ts % n.period;
          this->generated_latent_sequence[ts] +=
              n.spreads[partition] * norm_dist(this->rand_num_generator);
        }
    }
}


void AnalysisGraph::generate_head_node_latent_sequences(int samp, int num_timesteps) {
  for (int v : this->independent_nodes) {
    Node &n = (*this)[v];

    unordered_map<int, pair<double, double>> partition_mean_std;
    vector<double> change_medians;

    if (samp > -1) {
      partition_mean_std = this->latent_mean_std_collection[samp][v];
    }
    else {
      partition_mean_std = n.partition_mean_std;
      samp = 0;
    }

    this->generate_head_node_latent_sequence(v, num_timesteps, false, samp);
//    this->generate_head_node_latent_sequence_from_changes(n, num_timesteps, false);

    n.generated_latent_sequence.clear();
    n.generated_latent_sequence = this->generated_latent_sequence;
  }
}


void AnalysisGraph::update_head_node_latent_state_with_generated_derivatives(
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
    this->update_head_node_latent_state_with_generated_derivatives(
        ts, v, n.generated_latent_sequence);
  }
}
