# pragma once

#include "KDE.hpp"
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
  private:
  // The current β for this edge
  // TODO: Need to decide how to initialize this or
  // decide whether this is the correct way to do this.
  // β = tan(θ)
  // θ = atan(1) = Π/4
  // β = tan(atan(1)) = 1
  double theta = std::atan(1);
  bool frozen = false;

  public:
  std::string name;
  KDE kde;
  std::vector<Statement> evidence;

  std::vector<double> sampled_thetas;

  // The current log(p(θ))
  double logpdf_theta = 0;

  void freeze() {
      this->frozen = true;
  }

  void unfreeze() {
      this->frozen = false;
  }

  bool is_frozen() {
      return this->frozen;
  }

  void set_theta(double theta) {
      if (!this->frozen) {
          if (-M_PI_2 < theta && theta < M_PI_2) {
              this->theta = theta;
          }
          else if (-3 * M_PI_2 < theta && theta < -M_PI_2) {
              this->theta = theta + M_PI;
          }
          else if (M_PI_2 < theta && theta < 3 * M_PI_2) {
              this->theta = theta - M_PI;
          }
          else {
              std::cout << "\n\n\t**********ERROR: Edge.hpp - theta outside range coded for processing!!**********\n\n";
              std::cout << theta << std::endl;
          }
      }
  }

   double get_theta() const {
    return this->theta;
  }

  void compute_logpdf_theta() {
//    this->logpdf_theta = this->kde.logpdf(this->theta);
    this->logpdf_theta = this->kde.log_prior_hist[this->kde.theta_to_bin(this->theta)];
  }

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

