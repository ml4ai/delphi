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
  return j.dump();
}
