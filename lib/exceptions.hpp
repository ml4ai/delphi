#pragma once

class IndicatorNotFoundException : public std::exception {
  public:
  std::string msg;

  IndicatorNotFoundException(std::string msg) : msg(msg) {}

  const char* what() const throw() { return this->msg.c_str(); }
};

