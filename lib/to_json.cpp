#include "AnalysisGraph.hpp"
#include <nlohmann/json.hpp>

using namespace std;

string AnalysisGraph::to_json_string() {
  nlohmann::json j;
  j["id"] = this->id;
  vector<tuple<string, string, vector<double>>> data;
  for (auto e : this->edges()) {
    string source = (*this)[boost::source(e, this->graph)].name;
    string target = (*this)[boost::target(e, this->graph)].name;
    data.push_back({source, target, this->edge(e).kde.dataset});
  }
  j["edges"] = data;
  j["nodes"] = {};
  for (Node& n : this->nodes()) {
    // Just taking the first indicator for now, will try to incorporate multiple
    // indicators per node later.
    Indicator& indicator = n.indicators.at(0);
    j["indicatorValues"][n.name] = {{"indicator", indicator.name},
                                    {"mean", indicator.mean}};
  }
  return j.dump();
}
