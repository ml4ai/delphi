#pragma once

#include "kde.hpp"
#include "random_variables.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <string>

class IndicatorNotFoundException : public std::exception {
  public:
  std::string msg;

  IndicatorNotFoundException(std::string msg) : msg(msg) {}

  const char* what() const throw() { return this->msg.c_str(); }
};

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

  Indicator get_indicator(std::string indicator) {
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

    dbg("Extracting map_entry");
    auto map_entry = this->nameToIndexMap.extract(indicator_old);
    dbg("Map entry extracted");

    if (map_entry) // indicator_old is in the map
    {
      dbg("Map entry is true (?)");
      // Update the map entry and add the new indicator
      // in place of the old indicator
      map_entry.key() = indicator_new;
      dbg("");
      this->nameToIndexMap.insert(move(map_entry));
      dbg("");
      this->indicators[map_entry.mapped()] = Indicator(indicator_new, source);
      dbg("");
    }
    else // indicator_old is not attached to this node
    {
      std::cout << "Node::replace_indicator - indicator " << indicator_old
                << " is not attached to node " << name << std::endl;
      std::cout << "\tAdding indicator " << indicator_new << " afresh\n";
      dbg("");
      add_indicator(indicator_new, source);
      dbg("");
    }
  }

  // Utility function that clears the indicators vector and the name map.
  void clear_indicators() {
    this->indicators.clear();
    this->nameToIndexMap.clear();
  }
};

class Concept {
  public:
  std::string name;
  std::unordered_map<std::string, std::vector<std::tuple<std::string, double>>>
      db_refs;
};

enum class Polarity { positive = 1, negative = -1, unspecified };

class QualitativeDelta {
  public:
  Polarity polarity = Polarity::positive;
  std::vector<std::string> adjectives = {};
};

class Event {
  public:
  Concept concept;
  QualitativeDelta delta;

  std::string adjective;
  int polarity;
  std::string concept_name;

  Event(std::string adj, int pol, std::string con_name)
      : adjective{adj}, polarity{pol}, concept_name{con_name} {}

  Event(std::tuple<std::string, int, std::string> evnt) {
    adjective = std::get<0>(evnt);
    polarity = std::get<1>(evnt);
    concept_name = std::get<2>(evnt);
  }
};

class Statement {
  public:
  Event subject;
  Event object;

  Statement() : subject(Event("", 0, "")), object(Event("", 0, "")) {}

  Statement(Event sub, Event obj) : subject(sub), object(obj) {}
  int overall_polarity() {
    return this->subject.polarity * this->object.polarity;
  }
};

class Edge {
  public:
  std::string name;
  // TODO: Why kde is optional?
  // According to AnalysisGraph::construct_beta_pdfs()
  // it seems all the edges have a kde
  KDE kde;
  // std::vector<CausalFragment> causalFragments = {};

  std::vector<Statement> evidence;

  // The current β for this edge
  // TODO: Need to decide how to initialize this or
  // decide whethr this is the correct way to do this.
  double beta = 1.0;
  void change_polarity(int subject_polarity, int object_polarity) {
    for (Statement stmt : evidence) {
      stmt.subject.polarity = subject_polarity;
      stmt.object.polarity = object_polarity;
    }
  }


  double get_reinforcement() {
    std::vector<double> overall_polarities = {};
    for (auto stmt : this->evidence){
      overall_polarities.push_back(stmt.overall_polarity());
    }
    return delphi::utils::mean(overall_polarities);
  }
};

class GraphData {
  public:
  std::string name;
};

typedef boost::adjacency_list<boost::setS,
                              boost::vecS,
                              boost::bidirectionalS,
                              Node,
                              Edge,
                              GraphData>
    DiGraph;
