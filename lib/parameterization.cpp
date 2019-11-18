#include "AnalysisGraph.hpp"
#include "Node.hpp"
#include "Indicator.hpp"
#include "utils.hpp"
#include "data.hpp"

using namespace std;
using namespace delphi::utils;

void AnalysisGraph::parameterize(string country,
                                 string state,
                                 string county,
                                 int year,
                                 int month,
                                 map<string, string> units) {
  double stdev, mean;
  for (Node& node : this->nodes()) {
    for (Indicator& indicator : node.indicators) {
      if (in(units, indicator.name)) {
        indicator.set_unit(units[indicator.name]);
      }
      else {
        indicator.set_default_unit();
      }
      vector<double> data = get_data_value(indicator.name,
                                           country,
                                           state,
                                           county,
                                           year,
                                           month,
                                           indicator.unit,
                                           this->data_heuristic);

      mean = data.empty() ? 0 : delphi::utils::mean(data);
      indicator.set_mean(mean);
      stdev = 0.1 * abs(indicator.get_mean());
      stdev = stdev == 0 ? 1 : stdev;
      indicator.set_stdev(stdev);
    }
  }
}

