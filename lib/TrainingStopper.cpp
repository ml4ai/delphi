#include "Config.hpp"
#include "TrainingStopper.hpp"
#include <math.h>

TrainingStopper::TrainingStopper() {
  Config config;
  n_samples = config.get_int("training_stopping_n_samples", 200);
  delta_magnitude = config.get_double("training_stopping_min_delta", 0.1);
}


// return true if the training should stop
bool TrainingStopper::stop_training(std::vector<double> values, int end_idx){

  // if the end idx is less than the number of samples, do not stop training.
  if(end_idx < n_samples) {
    return false;
  }

  int start_idx = end_idx - n_samples;

  // find min and max between start and end indices
  double min = values[start_idx];
  double max = min;
  for(int i = start_idx+1; i < end_idx; i ++) {
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
