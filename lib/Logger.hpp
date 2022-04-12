#pragma once

#include <iostream>

#include <string>

using namespace std;

// Basic logfile writing
class Logger {
  public:
    Logger();
    ~Logger(){}
    void overwrite_logfile(string text);
    void info(string text);
    void warning(string text);
    void error(string text);
    string get_log_file_path();

  private:
    string timestamp();
    string filename = "logfile.txt";
    void write_to_logfile(string text, ios_base::openmode mode);
};
