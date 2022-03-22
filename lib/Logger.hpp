#pragma once

#include <iostream>

#include <string>

using namespace std;

// Basic logfile writing
class Logger {

  private:
    string timestamp();
    string filename = "../logfile.txt";
    void write_to_logfile(string text, ios_base::openmode mode);

  public:
    Logger();
    ~Logger(){}

    void overwrite_logfile(string text);
    void log_info(string label, string text);
    void log_warning(string label, string text);
    void log_error(string label, string text);
};
