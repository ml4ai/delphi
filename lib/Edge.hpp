# pragma once

#include "kde.hpp"
#include <string>
#include <vector>
#include "utils.hpp"

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
  KDE kde;
  std::vector<Statement> evidence;

  // The current β for this edge
  // TODO: Need to decide how to initialize this or
  // decide whether this is the correct way to do this.
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

