#pragma once

#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// Read config file for user-set parameters
class Config {
  public:
    Config();
    ~Config(){}

    int get_int(string field, int fallback);
    string get_string(string field, string fallback);
    double get_double(string field, double fallback); 

  private:
    json config;
    // this path presumes delphi running from the build directory.
    string filename = "../config.json";
    void read_config();
};
