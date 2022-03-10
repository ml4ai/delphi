#include "Config.hpp"
#include <nlohmann/json.hpp>

// Return the max samples for early stopping
int Config::get_early_stopping_n_samples() {
  update();
  return early_stopping_n_samples;
}

// Return the log liklihood delta for early stopping
double Config::get_early_stopping_log_liklihood_delta() {
  update();
  return early_stopping_log_liklihood_delta;
}

// read the config file and update our values
void update() {
}
