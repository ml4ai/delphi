#pragma once

#include <iostream>
#include <map>
#include "utils.hpp"
#include "Indicator.hpp"
#include "exceptions.hpp"
#include <limits.h>
#include "definitions.h"
#include <Eigen/Dense>

class Node {
  public:
  std::string name = "";
  double mean = 0;
  double std = 1;
  std::vector<double> generated_latent_sequence = {};
  int period = 1;
  DataAggregationLevel agg_level = DataAggregationLevel::MONTHLY;

  // Utilized only for head nodes with period > 1
  int tot_observations = 0;
  Eigen::VectorXd fourier_coefficients;
  Eigen::VectorXd best_fourier_coefficients;
  std::vector<double> fourier_freqs;
  double best_rmse = std::numeric_limits<double>::infinity();
  int n_components = 0;
  int best_n_components = 0;
  bool rmse_is_reducing = true;
  // Partition i refer to the midpoint between partitions i and (i+1) % period
  // Access:
  //  {partition --> [data value]}
  std::unordered_map<int, std::vector<double>> between_bin_midpoints;
  // Access:
  //  {partition --> ([time step], [data value])}
  std::unordered_map<int, std::pair<std::vector<int>, std::vector<double>>> partitioned_data = {};
  std::unordered_map<int, std::pair<std::vector<int>, std::vector<double>>> partitioned_absolute_change = {};
  std::unordered_map<int, std::pair<std::vector<int>, std::vector<double>>> partitioned_relative_change = {};
  std::unordered_map<int, std::pair<double, double>> partition_mean_std = {};
  std::vector<double> absolute_change_medians = {};
  std::vector<double> relative_change_medians = {};

  std::vector<double> centers = {};
  std::vector<double> spreads = {};
  std::vector<double> changes = {};
  std::vector<double> generated_latent_centers_for_a_period;
  std::vector<double> generated_latent_spreads_for_a_period;

  std::string center_measure = "mean"; // median or mean
  std::string model = "center"; // center, absolute_change, relative_change

  // Tracks whether bounds are available for this node
  bool has_max = false;
  bool has_min = false;
  // Latent space bounds
  double max_val = std::numeric_limits<double>::max();
  double min_val = std::numeric_limits<double>::min();
  // Observation space bounds for the first indicator attached to this node
  // latent bound = observation bound / scaling factor
  double max_val_obs = std::numeric_limits<double>::max();
  double min_val_obs = std::numeric_limits<double>::min();

  bool visited;
  LatentVar rv;
  std::string to_string() { return this->name; }

  std::vector<Indicator> indicators;
  // Maps each indicator name to its index in the indicators vector
  std::map<std::string, int> nameToIndexMap;

  int add_indicator(std::string indicator, std::string source) {
    // TODO: What if this indicator already exists?
    //      At the moment only the last indicator is recorded
    //      in the nameToIndexMap map
    // What if this indicator already exists?
    //*Loren: We just say it's already attached and do nothing.
    // As of right now, we are only attaching one indicator per node but even
    // if we were attaching multiple indicators to one node, I can't yet think
    // of a case where the numerical id (i.e. the order) matters. If we do come
    // across that case, we will just write a function that swaps ids.*
    if (delphi::utils::in(this->nameToIndexMap,indicator)) {
      std::cout << indicator << " already attached to " << name << std::endl;
      return -1;
    }

    this->nameToIndexMap[indicator] = this->indicators.size();
    this->indicators.push_back(Indicator(indicator, source));
    return this->indicators.size() - 1;
  }

  void delete_indicator(std::string indicator) {
    if (delphi::utils::in(this->nameToIndexMap, indicator)) {
      int ind_index = this->nameToIndexMap.at(indicator);
      this->nameToIndexMap.clear();
      this->indicators.erase(this->indicators.begin() + ind_index);
      // The values of the map object have to align with the vecter indexes.
      for (int i = 0; i < this->indicators.size(); i++) {
        this->nameToIndexMap.at(this->indicators[i].get_name()) = i;
      }
    }
    else {
      std::cout << "There is no indicator  " << indicator << "attached to "
                << name << std::endl;
    }
  }

  Indicator& get_indicator(std::string indicator) {
    try {
      return this->indicators[this->nameToIndexMap.at(indicator)];
    }
    catch (const std::out_of_range& oor) {
      throw IndicatorNotFoundException(indicator);
    }
  }

  void replace_indicator(std::string indicator_old,
                         std::string indicator_new,
                         std::string source) {

    auto map_entry = this->nameToIndexMap.extract(indicator_old);

    if (map_entry) // indicator_old is in the map
    {
      // Update the map entry and add the new indicator
      // in place of the old indicator
      map_entry.key() = indicator_new;
      this->nameToIndexMap.insert(move(map_entry));
      this->indicators[map_entry.mapped()] = Indicator(indicator_new, source);
    }
    else // indicator_old is not attached to this node
    {
      std::cout << "Node::replace_indicator - indicator " << indicator_old
                << " is not attached to node " << name << std::endl;
      std::cout << "\tAdding indicator " << indicator_new << " afresh\n";
      this->add_indicator(indicator_new, source);
    }
  }

  // Utility function that clears the indicators vector and the name map.
  void clear_indicators() {
    this->indicators.clear();
    this->nameToIndexMap.clear();
  }

  void print_indicators() {
    for (auto [ind, id] : this->nameToIndexMap) {
      std::cout << ind << " -> " << id << std::endl;
    }
    std::cout << std::endl;
  }

  void clear_state();

  void compute_bin_centers_and_spreads(const std::vector<int> &ts_sequence,
                                       const std::vector<double> &mean_sequence);

/**
 * Linear interpolate between bin midpoints. Midpoints are calculated only when
 * two consecutive modeling time steps has observations. Midpoints between bin b
 * and bin (b + 1) % period are assigned to midpoint bin b.
 * @param hn_id: ID of the head node where midpoints are being computed
 * @param ts_sequence: Modeling time step sequence where there are observations.
 * @param mean_sequence: Each modeling time step could have multiple
 *                       observations. When computing midpoints, we first
 *                       compute the average of multiple observations per
 *                       modeling time step and create a mean observation
 *                       sequence. We compute the midpoints between these
 *                       means.
 */
  void linear_interpolate_between_bin_midpoints(std::vector<int> &ts_sequence,
                                            std::vector<double> &mean_sequence);
};

