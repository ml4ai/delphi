#pragma once

#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// user-set parameters that can be adjusted in realtime
class Config {

  // this path presumes delphi running from the build directory.

  private:
    string filename = "../config.json";
    json read_config();
    void update();

  public:
    Config(){};
    Config(string filename):filename(filename){};
    ~Config(){}

    int get_early_stopping_n_samples();
    double get_early_stopping_log_liklihood_delta ();

    const string DEFAULT_FILENAME = "../config.json";
    const int DEFAULT_EARLY_STOPPING_N_SAMPLES = 100;
    const double DEFAULT_EARLY_STOPPING_LOG_LIKELIHOOD_DELTA = 0.0001;

    const string EARLY_STOPPING_LOG_LIKELIHOOD_DELTA = "log_liklihood_delta";
    const string EARLY_STOPPING_N_SAMPLES = "n_samples";
};
