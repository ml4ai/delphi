#pragma once

#include <string>
#include <vector>

std::vector<double> get_observations_for(std::string indicator,
                                         std::string country = "",
                                         std::string state = "",
                                         std::string county = "",
                                         int year = 2012,
                                         int month = 1,
                                         std::string unit = "",
                                         bool use_heuristic = false);

