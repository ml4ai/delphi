#include "Config.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;

Config::Config() {
  read_config(get_config_file_path());
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

// Determine config filename for our runtime environment
string Config::get_config_file_path() {

  // find the name of our current directory
  filesystem::path current_path = filesystem::current_path();
  filesystem::path current_path_filename = current_path.filename();

  // running in delphi/build (Linux or MAC OS)
  if(current_path_filename == "build") {
    return "../data/config.json";
  }

  // running in delphi (Docker container)
  if(current_path_filename == "delphi") {
    return "./data/config.json";
  }

  // unrecognized directory for delphi execution
  cerr << "Could not locate config file 'config.json'.  "
  << "Delphi should be run in the 'delphi/build' directory "
  << "on a Linux or MAC operating system, or in the 'delphi' directory "
  << "in a Docker container."
  << endl;

  exit(1);
}

// Return a JSON struct read from the config file
void Config::read_config(string filename) {
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
    exit(1);
  }
}
