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
    bool get_bool(string field, bool fallback);
    string get_config_file_path();

  private:
    json config;
    void read_config(string filename);
};
