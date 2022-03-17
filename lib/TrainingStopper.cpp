#include "Config.hpp"
#include "TrainingStopper.hpp"
#include <math.h>

TrainingStopper::TrainingStopper() {
  Config config;
  n_samples = config.get_training_stopping_sample_interval();
  delta_magnitude = config.get_training_stopping_min_log_likelihood_delta();
}


bool TrainingStopper::stop_training(std::vector<double> values, int end_idx){
  size_t start_idx = end_idx - n_samples;

  // we never stop before n_samples iterations.
  if(start_idx < 0) {
    return false;
  }

  // find min and max between start and end indices
  double min = values[start_idx];
  double max = min;
  for(size_t i = start_idx+1; i < end_idx; i ++) {
    double value = values[i];
    min = fmin(value, min);
    max = fmax(value, max);
  }

  // if the difference between min and max is within the delta magnitude,
  // then training should be stopped
  double delta_observed = abs(max-min);
  double delta_limit = abs(delta_magnitude);

  return (delta_observed <= delta_limit);
}
