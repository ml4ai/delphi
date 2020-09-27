/**
 * This is a temporary sandbox source file for Aishwarya to start developing
 * for Delphi. The methods coded here will be moved into appropriate places and
 * this file will be removed later.
 *
 * One motivation behind this file is to avoid merge conflicts as much as
 * possible since both Manujidnda and Aishwarya are going to develop on the
 * same branch.
 */

#include "AnalysisGraph.hpp"
#include "utils.hpp"
#include <fmt/format.h>
#include <fstream>
#include <range/v3/all.hpp>
#include <time.h>
#include <limits.h>

using namespace std;
using namespace delphi::utils;
using namespace fmt::literals;

// Just for debugging. Remove later
using fmt::print;
#include "dbg.h" // To quickly print the value of a variable v do dbg(v)


/*
 ============================================================================
 Private: Model serialization
 ============================================================================
*/

/**
 * TODO: Aishwarya, Implement the body of this function.
 * You can look into corresponding methods in causemose_integration.cpp
 * Suggestion: you can start by writing code to deserialize the json string
 * created by to_json_string() method in to_json.cpp
 * Also you can look at from_json_string()
 */
void AnalysisGraph::from_delphi_json_dict(const nlohmann::json &json_data) {
    this->id = json_data["id"];
    //this->name_to_vertex = json_data["concepts"];
    //this->ObservedStateSequence
     //nlohmann::json::parse  .items():  std::cout << jd.key() << " : " << jd.value() << '\n';

    for (auto& concept_name : json_data["concepts"])
    {
      this->add_node(concept_name);
    }

    //for (auto& concept_arr : json_data["conceptIndicators"])
    int conceptIndicators_arrSize = 0;
    if (sizeof(json_data["conceptIndicators"])){ int conceptIndicators_arrSize = sizeof(json_data["conceptIndicators"])/sizeof(json_data["conceptIndicators"][0]);}
    
    for (int v = 0; v < conceptIndicators_arrSize; v++)
    {
      Node &n = (*this)[v];
      for (auto& indicator_arr : json_data["conceptIndicators"][v])
      {
        int indicator_index =  n.add_indicator(indicator_arr["indicator"], indicator_arr["source"]); 
        n.indicators[indicator_index].aggregation_method = indicator_arr["func"];
        n.indicators[indicator_index].unit = indicator_arr["unit"];
      }
    }

    for (auto& edge_element : json_data["edges"])
    {
      for (auto& evidence : edge_element["evidence"])
      {
          auto subject = evidence.first;
          auto object = evidence.second;
          auto causal_fragment =
            CausalFragment({std::get<0>(subject), std::get<1>(subject), std::get<2>(subject)},
                           {std::get<0>(object), std::get<1>(object), std::get<2>(object)});
          this->add_edge(causal_fragment);
      }
      Edge* e = this->edge(edge_element["source"], edge_element["target"]);
      e.kde = edge_element["kernels"];
    }


    if (verbose) {
        this->training_range.first.first  = json_data["start_year"];
        this->training_range.first.second  = json_data["start_month"];
        this->training_range.second.first  = json_data["end_year"];
        this->training_range.second.second = json_data["end_month"];
    } else {
        // This is a pair of pairs where the first pair is <start_year,
        // start_month> and the second pair is <end_year, end_month>
        this->training_range = json_data["training_range"];
    }
    this->observed_state_sequence = json_data["observations"];


}

/*
 ============================================================================
 Public: Model serialization
 ============================================================================
*/

AnalysisGraph AnalysisGraph::deserialize_from_json_string(string json_string) {
  AnalysisGraph G;

  auto json_data = nlohmann::json::parse(json_string);
  G.from_delphi_json_dict(json_data);
  return G;
}

AnalysisGraph AnalysisGraph::deserialize_from_json_file(string filename) {
  AnalysisGraph G;

  auto json_data = load_json(filename);
  G.from_delphi_json_dict(json_data);
  return G;
}
