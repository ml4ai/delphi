#include "Logger.hpp"
#include <sys/time.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <limits.h>
#include <string.h>



using namespace std;

Logger::Logger() {
  filename = get_log_file_path();
} 

// Determine log filename for our runtime environment
string Logger::get_log_file_path() {

  // find the full path of our current working directory
  char cwd[PATH_MAX + 1]; 
  getcwd(cwd, sizeof(cwd));

  char* dirname = strrchr(cwd, '/');
  
  char full_path[PATH_MAX + 100];  

  // find the path to the config file based on OS

  // Docker 
  if(strcmp(dirname,"/delphi") == 0) {
    sprintf(full_path,"%s/data/%s", cwd, filename.c_str());
    return string(full_path);
  } 
  // Anything else
  if(strcmp(dirname,"/build") == 0) {
    sprintf(full_path,"%s/../data/%s", cwd, filename.c_str());
  } 

  // just drop it wherever
  return filename;
}

// return current time like this:  2022-02-17 14:33:52:016
string Logger::timestamp(){
  timeval curTime;
  gettimeofday(&curTime, NULL);
  int milli = curTime.tv_usec / 1000;
  char buffer [80];
  strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&curTime.tv_sec));
  char currentTime[100] = "";
  sprintf(currentTime, "%s:%03d ", buffer, milli);
  return string(currentTime);
}

// write the text to the logfile in the given mode
void Logger::write_to_logfile(string text, ios_base::openmode mode) {
  fstream file;
  file.open(filename, mode);
  if (file.is_open()) {
    file << timestamp() << text << endl;
    file.close();
  }
  else {
    cerr << "Could not write to logfile: " << filename << endl;
  }
}

// overwrite the logfile contents with this text
void Logger::overwrite_logfile(string text){
  write_to_logfile(text, ios_base::out);
}

// append this text to the logfile
void Logger::log_info(string label, string text){
  write_to_logfile(label + " INFO: " + text, ios_base::app);
}

// append this text to the logfile
void Logger::log_warning(string label, string text){
  write_to_logfile(label + " WARNING: " + text, ios_base::app);
}

// append this text to the logfile
void Logger::log_error(string label, string text){
  write_to_logfile(label + " ERROR: " + text, ios_base::app);
}
