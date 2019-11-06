#include "random_variables.hpp"
#include <iostream>
#include <iterator>
#include <random>

double RV::sample() {
  std::vector<double> s;

  std::sample(this->dataset.begin(),
              this->dataset.end(),
              std::back_inserter(s),
              1,
              std::mt19937{std::random_device{}()});

  return s[0];
}
