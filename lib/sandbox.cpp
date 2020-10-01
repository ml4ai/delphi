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
typedef pair<tuple<string, int, string>, tuple<string, int, string>> Evidence_Pair;
typedef tuple<int, int, vector<double>, vector<pair<tuple<string, int, string>, tuple<string, int, string>>>> Edge_tuple;

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


    int conceptIndicators_arrSize = 0;
    if (sizeof(json_data["conceptIndicators"])){ int conceptIndicators_arrSize = sizeof(json_data["conceptIndicators"])/sizeof(json_data["conceptIndicators"][0]);}
    
    //for (int v = 0; v < conceptIndicators_arrSize; v++)
    //{
    //  Node &n = (*this)[v];

    if (verbose){
      for (auto& concept_arr : json_data["concepts"])
      {
        this->add_node(concept_arr[0][1]);
      }
    } else{
      for (auto& concept_name : json_data["concepts"])
      {
        this->add_node(concept_name);
      }
    }

    for (Node &n : this->nodes())
    {    
  
      for (auto indicator_arr : json_data["conceptIndicators"][this->name_to_vertex.at(n.name)])
      {
        if (verbose){
          int indicator_index =  n.add_indicator(indicator_arr[0][1], indicator_arr[2][1]); 
          n.indicators[indicator_index].aggregation_method = indicator_arr[3][1];
          n.indicators[indicator_index].unit = indicator_arr[4][1];
        } 
        else{
            /*
          int indicator_index = 0;
          if (sizeof(indicator_arr)){ int indicator_index = sizeof(indicator_arr)/sizeof(indicator_arr[0]);}
    
          for (int i = 0; i < indicator_index; i++){
            n.add_indicator(indicator_arr[i][0][1], indicator_arr[i][1][1]); 
            n.indicators[i].aggregation_method = indicator_arr[i][2][1];
            n.indicators[i].unit = indicator_arr[i][3][1];
          }
          */
        }
      }
    }
    // Manujinda solution start
    if (!verbose) {
        for (int v = 0; v < this->num_vertices(); v++) {
            Node &n = (*this)[v];

            auto ind_data = json_data["conceptIndicators"][v];

            for (auto ind : ind_data) {
                string ind_name = ind["indicator"].get<string>();
                n.add_indicator(ind_name, ind["source"].get<string>());
                n.get_indicator(ind_name).set_aggregation_method(ind["func"].get<string>());
                n.get_indicator(ind_name).set_unit(ind["unit"].get<string>());
            }
        }
    }
    // Manujinda solution end
    

    if (verbose) {
      for (auto& edge_element : json_data["edges"])
      {
        for (Evidence evidence : edge_element[0]["evidence"])
        {
          for (Evidence_Pair evidence_pair : evidence){
            tuple<string, int, string> subject = evidence_pair.first;
            tuple<string, int, string> object = evidence_pair.second;
            CausalFragment causal_fragment =
              CausalFragment({get<0>(subject), get<1>(subject), get<2>(subject)},
                             {get<0>(object), get<1>(object), get<2>(object)});
            this->add_edge(causal_fragment);
          }
        }
        string source = edge_element["source"].get<string>();
        string target = edge_element["target"].get<string>();
        this->edge(source, target).kde.dataset = edge_element["kernels"].get<vector<double>>();
      }
    }
    else{
      for (Edge_tuple edge_element : json_data["edges"])
      {
        for (Evidence_Pair evidence : get<3>(edge_element))
        {
            tuple<string, int, string> subject = evidence.first;
            tuple<string, int, string> object = evidence.second;
            CausalFragment causal_fragment =
              CausalFragment({get<0>(subject), get<1>(subject), get<2>(subject)},
                             {get<0>(object), get<1>(object), get<2>(object)});
            this->add_edge(causal_fragment);
        }
        //print("this:  {0} {1}", get<0>(edge_element), get<1>(edge_element));
        this->edge(get<0>(edge_element), get<1>(edge_element)).kde.dataset = get<2>(edge_element);
      }
    }


    if (verbose) {
        this->training_range.first.first  = json_data["start_year"];
        this->training_range.first.second  = json_data["start_month"];
        this->training_range.second.first  = json_data["end_year"];
        this->training_range.second.second = json_data["end_month"];
    } else {
        this->training_range = json_data["training_range"];
    }

    this->observed_state_sequence = json_data["observations"].get<ObservedStateSequence>();
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
