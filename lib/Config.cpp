#include "Config.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

// This configuration class reads the config file each time
// a variable is retrieved.

// Return the max samples for early stopping
int Config::get_early_stopping_n_samples() {
  json config = read_config();
  if(config.contains(EARLY_STOPPING_N_SAMPLES)){
    return (int)config[EARLY_STOPPING_N_SAMPLES];
  }
  return DEFAULT_EARLY_STOPPING_N_SAMPLES;
}

// Return the log liklihood delta for early stopping
double Config::get_early_stopping_log_liklihood_delta() {
  json config = read_config();
  if(config.contains(EARLY_STOPPING_LOG_LIKELIHOOD_DELTA)){
    return (double)config[EARLY_STOPPING_LOG_LIKELIHOOD_DELTA];
  }
  return DEFAULT_EARLY_STOPPING_LOG_LIKELIHOOD_DELTA;
}

// Return a JSON struct read from the config file
json Config::read_config() {
  ifstream file;
  file.open(filename);
  string jsonString;
  string line;
  json config;
  if ( file.is_open() ) {
    while ( file ) { 
      getline (file, line);
      jsonString += line;
    }
    file.close();
    config = json::parse(jsonString);
  }
  else {
    cerr << "Couldn't open config file at '" << filename << "'" << endl;
  }
  return config;
}

