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
    int training_stopping_sample_interval = DEFAULT_TRAINING_STOPPING_SAMPLE_INTERVAL;
    int training_burn = DEFAULT_TRAINING_BURN;
    double training_stopping_min_log_liklihood_delta = 
      DEFAULT_TRAINING_STOPPING_MIN_LOG_LIKELIHOOD_DELTA;

  public:
    Config();
    Config(string filename);
    ~Config(){}

    int get_training_stopping_sample_interval();
    int get_training_burn();
    double get_training_stopping_min_log_liklihood_delta ();

    const string DEFAULT_FILENAME = "../config.json";
    const int DEFAULT_TRAINING_STOPPING_SAMPLE_INTERVAL = 200;
    const int DEFAULT_TRAINING_BURN = 1000;
    const double DEFAULT_TRAINING_STOPPING_MIN_LOG_LIKELIHOOD_DELTA = 0.0;

    const string JSON_TRAINING_STOPPING_MIN_LOG_LIKELIHOOD_DELTA = 
      "training_stopping_min_log_likelihood_delta";
    const string JSON_TRAINING_STOPPING_SAMPLE_INTERVAL = 
      "training_stopping_sample_interval";
    const string JSON_TRAINING_BURN = "training_burn";
};
