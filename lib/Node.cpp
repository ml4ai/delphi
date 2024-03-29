#include "Node.hpp"

using namespace std;

void Node::clear_state()
{
    this->generated_latent_sequence.clear();
    this->tot_observations = 0;
    this->fourier_coefficients.resize(0);
    this->best_fourier_coefficients.resize(0);
    this->fourier_freqs.clear();
    this->best_rmse = std::numeric_limits<double>::infinity();
    this->n_components = 0;
    this->best_n_components = 0;
    this->rmse_is_reducing = true;
    this->between_bin_midpoints.clear();
    this->partitioned_data.clear();
    this->partitioned_absolute_change.clear();
    this->partitioned_relative_change.clear();
    this->partition_mean_std.clear();
    this->absolute_change_medians.clear();
    this->relative_change_medians.clear();
    this->centers.clear();
    this->spreads.clear();
    this->changes.clear();
    this->generated_latent_centers_for_a_period.clear();
    this->generated_latent_spreads_for_a_period.clear();
}

void Node::compute_bin_centers_and_spreads(const vector<int> &ts_sequence,
                                           const vector<double> &mean_sequence) {
    double center;
    vector<int> filled_observation_timesteps_within_a_period;
    this->centers = vector<double>(this->period + 1);
    this->spreads = vector<double>(this->period);
    this->generated_latent_centers_for_a_period = vector<double>(this->period, 0);
    this->generated_latent_spreads_for_a_period = vector<double>(this->period, 0);

    for (const auto & [ partition, data ] : this->partitioned_data) {
        if (this->center_measure.compare("mean") == 0) {
            center = delphi::utils::mean(data.second);
        } else {
            center = delphi::utils::median(data.second);
        }
        this->centers[partition] = center;

        double spread = 0;
        if (data.second.size() > 1) {
            if (this->center_measure.compare("mean") == 0) {
                spread = delphi::utils::standard_deviation(center,
                                                           data.second);
            } else {
                spread = delphi::utils::median_absolute_deviation(center,
                                                                  data.second);
            }
        }
        this->spreads[partition] = spread;

        if (!data.first.empty()) {
            this->generated_latent_centers_for_a_period[partition]
                = center;
            this->generated_latent_spreads_for_a_period[partition]
                = spread;
            filled_observation_timesteps_within_a_period.push_back(
                partition);
        }
    }

    sort(filled_observation_timesteps_within_a_period.begin(),
         filled_observation_timesteps_within_a_period.end());

    // Linear interpolate values for the empty bins within a period
    if (filled_observation_timesteps_within_a_period.size() > 1) {
        // There are more than one bin with data. We could linear interpolate
        // between them.

        for (int i = 0; i < filled_observation_timesteps_within_a_period.size();
                                                                          i++) {
            int observation_timestep_within_a_period_start =
                filled_observation_timesteps_within_a_period[i];
            int observation_timestep_within_a_period_end =
                filled_observation_timesteps_within_a_period
                    [(i + 1) %
                     filled_observation_timesteps_within_a_period.size()];

            // Compute the number of empty bins between two consecutive bins
            // with data.
            int num_missing_observation_timesteps = 0;
            if (observation_timestep_within_a_period_end >
                                   observation_timestep_within_a_period_start) {
                num_missing_observation_timesteps =
                                 observation_timestep_within_a_period_end -
                                 observation_timestep_within_a_period_start - 1;
            }
            else {
                num_missing_observation_timesteps =
                                  (this->period - 1 -
                                   observation_timestep_within_a_period_start) +
                                   observation_timestep_within_a_period_end;
            }

            // Linear interpolate centers and spreads for empty bins
            for (int missing_observation_timestep = 1;
                 missing_observation_timestep <= num_missing_observation_timesteps;
                 missing_observation_timestep++) {

                this->generated_latent_centers_for_a_period
                    [(observation_timestep_within_a_period_start +
                      missing_observation_timestep) % this->period] =
                    ((num_missing_observation_timesteps -
                      missing_observation_timestep + 1) *
                         this->generated_latent_centers_for_a_period
                             [observation_timestep_within_a_period_start] +
                     (missing_observation_timestep) *
                     this->generated_latent_centers_for_a_period
                         [observation_timestep_within_a_period_end]) /
                    (num_missing_observation_timesteps + 1);

                this->generated_latent_spreads_for_a_period
                    [(observation_timestep_within_a_period_start +
                      missing_observation_timestep) % this->period] =
                    ((num_missing_observation_timesteps -
                      missing_observation_timestep + 1) *
                         this->generated_latent_spreads_for_a_period
                             [observation_timestep_within_a_period_start] +
                     (missing_observation_timestep) *
                     this->generated_latent_spreads_for_a_period
                         [observation_timestep_within_a_period_end]) /
                    (num_missing_observation_timesteps + 1);
            }
        }
    } else if (filled_observation_timesteps_within_a_period.size() == 1) {
        // There is only one bin with data. Copy the center and spread of that
        // bin to all the other bins.
        for (int observation_timestep = 0;
             observation_timestep < this->generated_latent_centers_for_a_period.size();
             observation_timestep++) {
            this->generated_latent_centers_for_a_period[observation_timestep] =
                this->generated_latent_centers_for_a_period
                    [filled_observation_timesteps_within_a_period[0]];
            this->generated_latent_spreads_for_a_period[observation_timestep] =
                this->generated_latent_spreads_for_a_period
                    [filled_observation_timesteps_within_a_period[0]];
        }
    }

    this->changes = vector<double>(this->centers.size(), 0.0);
    this->centers[this->period] = this->centers[0];
    if (this->model.compare("center") != 0) {
        // model == absolute_change
        adjacent_difference(this->centers.begin(), this->centers.end(),
                            this->changes.begin());
        if (this->model.compare("relative_change") == 0) {
            transform(this->centers.begin(),
                      this->centers.end() - 1,
                      this->changes.begin() + 1,
                      this->changes.begin() + 1,
                      [&](double start_value, double abs_change) {
                          return abs_change / (start_value + 1);
                      });
        }
    }

    // Experiment: First calculate adjacent changes, then partition changes
    // and compute the center of each changes partition
    /*
    // Absolute changes
    vector<double> absolute_change = vector<double>(mean_sequence.size());
    adjacent_difference(mean_sequence.begin(),
                        mean_sequence.end(),
                        absolute_change.begin());

    // Relative changes
    vector<double> relative_change = vector<double>(mean_sequence.size() - 1);
    transform(mean_sequence.begin(), mean_sequence.end() - 1,
              absolute_change.begin() + 1, relative_change.begin(),
              [&](double start_value, double abs_change)
              {return abs_change / (start_value + 1);});

    // Partition changes
    for (int ts = 0; ts < relative_change.size(); ts++) {
        int partition = ts % this->period;
        this->partitioned_absolute_change[partition].first.push_back(ts_sequence[ts]);
        this->partitioned_absolute_change[partition].second.push_back(absolute_change[ts + 1]);

        this->partitioned_relative_change[partition].first.push_back(ts_sequence[ts]);
        this->partitioned_relative_change[partition].second.push_back(relative_change[ts]);
    }

    // Compute partition centers
    //this->changes = vector<double>(this->period + 1);
    for (const auto & [ partition, data ] : this->partitioned_absolute_change) {
        double partition_median = delphi::utils::median(data.second);
        this->changes[partition + 1] = partition_median;
    }
    //for (const auto & [ partition, data ] : this->partitioned_relative_change) {
    //    double partition_median = delphi::utils::median(data.second);
    //    this->changes[partition + 1] = partition_median;
    //}

    // Experimenting with zero centering the centers
    //vector<double> only_changes = vector<double>(this->changes.begin() + 1, this->changes.end());
    //double change_mean = delphi::utils::mean(only_changes);
    //transform(this->changes.begin() + 1, this->changes.end(),
    //          this->changes.begin() + 1,
    //          [&](double val){return val - change_mean;});
    */
}

/**
 * Linear interpolate between bin midpoints. Midpoints are calculated only when
 * two consecutive modeling time steps has observations. Midpoints between bin b
 * and bin (b + 1) % period are assigned to midpoint bin b.
 * @param hn_id: ID of the head node where midpoints are being computed
 * @param ts_sequence: Modeling time step sequence where there are observations.
 * @param mean_sequence: Each modeling time step could have multiple
 *                       observations. When computing midpoints, we first
 *                       compute the average of multiple observations per
 *                       modeling time step and create a mean observation
 *                       sequence. We compute the midpoints between these
 *                       means.
 */
void Node::linear_interpolate_between_bin_midpoints(
                                           std::vector<int> &ts_sequence,
                                           std::vector<double> &mean_sequence) {
    for (int mean_seq_idx = 0; mean_seq_idx < ts_sequence.size() - 1;
         mean_seq_idx++) {
        if (ts_sequence[mean_seq_idx] == ts_sequence[mean_seq_idx + 1] - 1) {
            // We have two consecutive time points with observations. So, we
            // could compute a between bin midpoint. We place the midpoints
            // between bin b and bin (b+1) % period in midpoint bin b.
            int partition = ts_sequence[mean_seq_idx] % this->period;
            this->between_bin_midpoints[partition].push_back(
                                       (mean_sequence[mean_seq_idx] +
                                        mean_sequence[mean_seq_idx + 1]) / 2.0);
        }
    }
}

