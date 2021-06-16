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

void AnalysisGraph::generate_head_node_latent_sequence(int period,
                                                       vector<double> centers,
                                                       vector<double> spreads,
                                                       int num_timesteps,
                                                       bool sample) {
    this->generated_latent_sequence = vector<double>(num_timesteps);

    for (int ts = 0; ts < num_timesteps; ts++) {
      int partition = ts % period;

      if (sample) {
          this->generated_latent_sequence[ts] = centers[partition]
                                                 + spreads[partition]
                                                 * norm_dist(this->rand_num_generator);
      } else {
          this->generated_latent_sequence[ts] = centers[partition];
      }
    }
}

void AnalysisGraph::generate_head_node_latent_sequence_from_changes(int period,
                                                                    double initial_value,
                                                                    vector<double> spreads,
                                                                    vector<double> changes,
                                                                    string model,
                                                                    int num_timesteps,
                                                                    bool sample) {
    this->generated_latent_sequence = vector<double>(num_timesteps);

    this->generated_latent_sequence[0] = initial_value;

    for (int ts = 0; ts < num_timesteps - 1; ts++) {
        int partition = ts % period;

        if (model.compare("absolute_change") == 0) {
            this->generated_latent_sequence[ts + 1] =
                this->generated_latent_sequence[ts] + changes[partition + 1];
        } else if (model.compare("relative_change") == 0) {
            this->generated_latent_sequence[ts + 1] =
                this->generated_latent_sequence[ts] + changes[partition + 1]
                                                        * (this->generated_latent_sequence[ts] + 1);
        }
    }

    if (sample) {
        for (int ts = 0; ts < num_timesteps; ts++) {
          int partition = ts % period;
          this->generated_latent_sequence[ts] +=
              spreads[partition] * norm_dist(this->rand_num_generator);
        }
    }
}

void AnalysisGraph::generate_from_data_mean_and_std_gussian(
    double mean,
    double std,
    int num_timesteps,
    unordered_map<int, pair<double, double>> partition_mean_std,
    int period) {
  this->generated_latent_sequence = vector<double>(num_timesteps);

  for (int ts = 0; ts < num_timesteps; ts++) {
    int partition = ts % period;
    this->generated_latent_sequence[ts] = partition_mean_std[partition].first;
//                                       + partition_mean_std[partition].second
//                                       * norm_dist(this->rand_num_generator);
  }
}

void AnalysisGraph::generate_from_absolute_change(
    int num_timesteps,
    double initial_value,
    vector<double> absolute_change_medians,
    int period) {
  this->generated_latent_sequence = vector<double>(num_timesteps);

  this->generated_latent_sequence[0] = initial_value;

  for (int ts = 0; ts < num_timesteps - 1; ts++) {
    int partition = ts % period;
    this->generated_latent_sequence[ts + 1] = this->generated_latent_sequence[ts] + absolute_change_medians[partition];
  }
}

void AnalysisGraph::generate_from_relative_change(
    int num_timesteps,
    double initial_value,
    vector<double> absolute_change_medians,
    int period) {
  this->generated_latent_sequence = vector<double>(num_timesteps);

  this->generated_latent_sequence[0] = 1;

  for (int ts = 1; ts < num_timesteps; ts++) {
    int partition = ts % period;
    this->generated_latent_sequence[ts] = this->generated_latent_sequence[ts - 1] +
                                         (this->generated_latent_sequence[ts - 1] + 1) *
                                              absolute_change_medians[partition];
  }
}

void AnalysisGraph::generate_independent_node_latent_sequences(int samp, int num_timesteps) {
  for (int v : this->independent_nodes) {
    Node &n = (*this)[v];

    double mean;
    double std;
    unordered_map<int, pair<double, double>> partition_mean_std;
//    unordered_map<int, double> change_medians;
    vector<double> change_medians;

    if (samp > -1) {
      mean = this->latent_mean_collection[samp][v];
      std = this->latent_std_collection[samp][v];
      partition_mean_std = this->latent_mean_std_collection[samp][v];
    }
    else {
      mean = n.mean;
      std = n.std;
      partition_mean_std = n.partition_mean_std;
    }

//    change_medians = n.absolute_change_medians;
//    this->generate_from_absolute_change(
//        num_timesteps, n.partition_mean_std[0].first, change_medians, n.period);


//    this->generate_head_node_latent_sequence(n.period, n.centers, n.spreads,
//                                             n.changes, num_timesteps, false);
    this->generate_head_node_latent_sequence_from_changes(n.period, n.centers[0],
                                                  n.spreads, n.changes, n.model,
                                                  num_timesteps, false);


    n.generated_latent_sequence.clear();
    n.generated_latent_sequence = this->generated_latent_sequence;

//    this->generate_from_data_mean_and_std_gussian(
//        mean, std, num_timesteps, partition_mean_std, n.period);

//    vector<double> difference = vector<double>(this->generated_latent_sequence.size());
//    transform(n.generated_latent_sequence.begin(), n.generated_latent_sequence.end(),
//              this->generated_latent_sequence.begin(), difference.begin(),
//              [&](double abs_change, double value){return abs_change - value;});
//    dbg(n.name);
//    dbg(this->generated_latent_sequence);
//    dbg(n.generated_latent_sequence);
//    dbg(difference);


//    change_medians = n.relative_change_medians;
//    this->generate_from_relative_change(
//        num_timesteps, n.partition_mean_std[0].first, change_medians, n.period);
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
