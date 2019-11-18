#pragma once

#include <algorithm>
#include <string>
#include <vector>

class RV {
public:
  std::string name;
  double value;
  std::vector<double> dataset;
  RV(std::string name) : name(name) {}
  double sample();
};

class LatentVar : public RV {
public:
  double partial_t;

  LatentVar() : RV("") {}
  LatentVar(std::string name) : RV(name) {}
};
