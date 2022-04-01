#include "Config.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <limits.h>
#include <string.h>

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

// get a boolean val
bool Config::get_bool(string field, bool fallback) {
  return config.contains(field) ? (bool)config[field] : fallback;
}

// Determine config filename for our runtime environment
string Config::get_config_file_path() {

  // find the full path of our current working directory
  char cwd[PATH_MAX + 1];
  getcwd(cwd, sizeof(cwd));

  char* dirname = strrchr(cwd, '/');
  
  char filename[PATH_MAX + 100];  

  // find the path to the config file based on OS
  if(strcmp(dirname,"/delphi") == 0) {
    // Docker 
    sprintf(filename,"%s/data/%s", cwd, FILENAME.c_str());
  } else if(strcmp(dirname,"/build") == 0) {
    // Anything else
    sprintf(filename,"%s/../data/%s", cwd, FILENAME.c_str());
  }

  // check that the file actually exists
  if( access(filename, F_OK ) == 0 ) {
    // file exists
    return string(filename);
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
