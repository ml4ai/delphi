#include "Config.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

Config::Config() {
  init();
}

Config::Config(string filename):filename(filename){
  init();
}

void Config::init() {
  json config = read_config();

  if(config.contains(TRAINING_MIN_LOG_LIKELIHOOD_DELTA)){
    training_min_log_liklihood_delta = 
      config[TRAINING_MIN_LOG_LIKELIHOOD_DELTA];
  }

  if(config.contains(TRAINING_N_SAMPLES)){
    training_n_samples = config[TRAINING_N_SAMPLES];
  }
}

// Return the max samples for training
int Config::get_training_n_samples() {
  return training_n_samples;
}

// Return the minimum log liklihood delta for early stopping
double Config::get_training_min_log_liklihood_delta() {
  return training_min_log_liklihood_delta;
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

