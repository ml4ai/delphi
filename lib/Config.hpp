#pragma once

#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// Read config file for user-set parameters
class Config {

  // this path presumes delphi running from the build directory.

  private:
    void init();
    string filename = "../config.json";
    json read_config();
    void update();
    int training_n_samples = DEFAULT_TRAINING_N_SAMPLES;
    double training_min_log_liklihood_delta = 
	    DEFAULT_TRAINING_MIN_LOG_LIKELIHOOD_DELTA;

  public:
    Config();
    Config(string filename);
    ~Config(){}

    int get_training_n_samples();
    double get_training_min_log_liklihood_delta ();

    const string DEFAULT_FILENAME = "../config.json";
    const int DEFAULT_TRAINING_N_SAMPLES = 100;
    const double DEFAULT_TRAINING_MIN_LOG_LIKELIHOOD_DELTA = 0.0001;

    const string TRAINING_MIN_LOG_LIKELIHOOD_DELTA = 
      "early_stopping_min_log_likelihood_delta";
    const string TRAINING_N_SAMPLES = "early_stopping_n_samples";
};
