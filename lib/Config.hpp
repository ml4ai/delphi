#pragma once

#include "Config.hpp"
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// user-set parameters that can be adjusted in realtime
class Config {

  private:
    json config;
    void update();
    double early_stopping_n_samples = 100;
    double early_stopping_log_liklihood_delta = 0.001;

  public:
    Config(){}
    ~Config(){}

    int get_early_stopping_n_samples();
    double get_early_stopping_log_liklihood_delta ();
};
