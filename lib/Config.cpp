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

  if(config.contains(JSON_TRAINING_STOPPING_MIN_LOG_LIKELIHOOD_DELTA)){
    training_stopping_min_log_liklihood_delta = 
      config[JSON_TRAINING_STOPPING_MIN_LOG_LIKELIHOOD_DELTA];
  }

  if(config.contains(JSON_TRAINING_STOPPING_SAMPLE_INTERVAL)){
    training_stopping_sample_interval = config[JSON_TRAINING_STOPPING_SAMPLE_INTERVAL];
  }

  if(config.contains(JSON_TRAINING_BURN)){
    training_burn = config[JSON_TRAINING_BURN];
  }
}

// How many samples to generate
int Config::get_training_burn() {
  return training_burn;
}

// Sample interval over which to check for variance in log likelihood
int Config::get_training_stopping_sample_interval() {
  return training_stopping_sample_interval;
}

// Minimum log liklihood delta for early stopping
double Config::get_training_stopping_min_log_liklihood_delta() {
  return training_stopping_min_log_liklihood_delta;
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
