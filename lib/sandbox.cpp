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


typedef vector<pair<tuple<string, int, string>, tuple<string, int, string>>>
Evidence;
//typedef vector<vector<string, int>, vector<string, int>, vector<string, Edge::KDE>, vector<string, Evidence>> Edges_vector;

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
void AnalysisGraph::from_delphi_json_dict(const nlohmann::json &json_data, bool verbose) {
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
      for (auto indicator_arr : json_data["conceptIndicators"][v])
      {
        int indicator_index =  n.add_indicator(indicator_arr["indicator"], indicator_arr["source"]); 
        n.indicators[indicator_index].aggregation_method = indicator_arr["func"];
        n.indicators[indicator_index].unit = indicator_arr["unit"];
      }
    }

    for (auto& edge_element : json_data["edges"])
    {
      if (verbose) {
        //for (auto& element : edge_element[0]["evidence"].items())
        for (auto& element : edge_element[0].items())
        {
          if (element.key() == "evidence"){
            cout << element.value() << endl;
            auto evidence = element.value();
            auto subject = evidence.first;
            auto object = evidence.second;
            auto causal_fragment =
              CausalFragment({get<0>(subject), get<1>(subject), get<2>(subject)},
                             {get<0>(object), get<1>(object), get<2>(object)});
            this->add_edge(causal_fragment);
          }
        }

        string source = edge_element[2]["source"].get<string>();
        string target = edge_element[3]["target"].get<string>();
        this->edge(source, target).kde.dataset = edge_element[1]["kernels"].get<vector<double>>();
      
        //for (auto evidence : edge_element[0]["evidence"])
        //{
        //    auto subject = evidence.first;
        //    auto object = evidence.second;
        //    auto causal_fragment =
        //      CausalFragment({get<0>(subject), get<1>(subject), get<2>(subject)},
        //                     {get<0>(object), get<1>(object), get<2>(object)});
        //    this->add_edge(causal_fragment);
        //}
        ////auto e = this->edge(edge_element["source"], edge_element["target"]);
        ////e->kde = edge_element["kernels"];
        //string source = edge_element["source"].get<string>();
        //string target = edge_element["target"].get<string>();
        //this->edge(source, target).kde.dataset = edge_element["kernels"].get<vector<double>>();
      }
      else{
        //int source;
        //int target;
        //KDE kde_data;
        //Evidence evidences;
//
        //tie(source,target,kde_data,evidences) = edge_element; 
        //for (auto evidence : evidences)
        //{
        //    auto subject = evidence.first;
        //    auto object = evidence.second;
        //    auto causal_fragment =
        //      CausalFragment({get<0>(subject), get<1>(subject), get<2>(subject)},
        //                     {get<0>(object), get<1>(object), get<2>(object)});
        //    this->add_edge(causal_fragment);
        //}
        //auto e = this->edge(source, target);
        ////this->KDE::edge(source, target).kde(kde_data);



        for (auto evidence : get<3>(edge_element))
        {
            auto subject = evidence.first;
            auto object = evidence.second;
            auto causal_fragment =
              CausalFragment({get<0>(subject), get<1>(subject), get<2>(subject)},
                             {get<0>(object), get<1>(object), get<2>(object)});
            this->add_edge(causal_fragment);
        }
        auto e = this->edge(get<0>(edge_element), get<1>(edge_element));
        e->kde = get<2>(edge_element);
      }
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
    //////////////this->observed_state_sequence = json_data["observations"].get<vector<vector<pair<int, int>>>>();


}

/*
 ============================================================================
 Public: Model serialization
 ============================================================================
*/

AnalysisGraph AnalysisGraph::deserialize_from_json_string(string json_string, bool verbose) {
  AnalysisGraph G;

  nlohmann::json json_data = nlohmann::json::parse(json_string);
  G.from_delphi_json_dict(json_data, verbose);
  return G;
}

AnalysisGraph AnalysisGraph::deserialize_from_json_file(string filename, bool verbose) {
  AnalysisGraph G;

  nlohmann::json json_data = load_json(filename);
  G.from_delphi_json_dict(json_data, verbose);
  return G;
}
