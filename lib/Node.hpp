#pragma once

#include <iostream>
#include <map>
#include "utils.hpp"
#include "Indicator.hpp"
#include "exceptions.hpp"

class Node {
  public:
  std::string name = "";
  bool visited;
  LatentVar rv;
  std::string to_string() { return this->name; }

  std::vector<Indicator> indicators;
  // Maps each indicator name to its index in the indicators vector
  std::map<std::string, int> nameToIndexMap;

  void add_indicator(std::string indicator, std::string source) {
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
      return;
    }

    this->nameToIndexMap[indicator] = this->indicators.size();
    this->indicators.push_back(Indicator(indicator, source));
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
};

