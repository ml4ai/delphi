#include "AnalysisGraph.hpp"
#include <fmt/format.h>
#include <nlohmann/json.hpp>

using namespace std;
using namespace fmt::literals;

string AnalysisGraph::to_json_string(int indent) {
  nlohmann::json j;
  j["id"] = this->id;
  vector<tuple<string, string, vector<double>>> data;
  for (auto e : this->edges()) {
    string source = (*this)[boost::source(e, this->graph)].name;
    string target = (*this)[boost::target(e, this->graph)].name;
    data.push_back({source, target, this->edge(e).kde.dataset});
  }
  j["edges"] = data;
  for (Node& n : this->nodes()) {
    // Just taking the first indicator for now, will try to incorporate multiple
    // indicators per node later.
    if (n.indicators.size() == 0) {
      throw runtime_error("Node {} has no indicators!"_format(n.name));
    }
    Indicator& indicator = n.indicators.at(0);
    j["indicatorData"][n.name] = {
        {"name", indicator.name},
        {"mean", indicator.mean},
        {"source", indicator.source},
    };
  }
  return j.dump(indent);
}
