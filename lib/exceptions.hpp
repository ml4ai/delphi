#pragma once

class IndicatorNotFoundException : public std::exception {
  public:
  std::string msg;

  IndicatorNotFoundException(std::string msg) : msg(msg) {}

  const char* what() const throw() { return this->msg.c_str(); }
};

class BadCausemosInputException : public std::exception {
  public:
  std::string msg;

  BadCausemosInputException(std::string msg) : msg(msg) {}
  BadCausemosInputException() : msg("Bad CauseMos Input") {}

  const char* what() const throw() { return this->msg.c_str(); }
};

