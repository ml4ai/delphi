#include "Config.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

Config::Config() {
  read_config();
}

// get an int val
int Config::get_int(string field, int fallback) {
  return config.contains(field) ? (int)config[field] : fallback;
}

// get a string val
string Config::get_string(string field, string fallback) {
  return config.contains(field) ? (string)config[field] : fallback;
}

// get a double val
double Config::get_double(string field, double fallback) {
  return config.contains(field) ? (double)config[field] : fallback;
}

// Return a JSON struct read from the config file
void Config::read_config() {
  ifstream file;
  file.open(filename);
  string jsonString;
  string line;
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
}
