// Usage: ./create_model <path_to_JSON_file>

#include "AnalysisGraph.hpp"
#include <boost/program_options.hpp>
using namespace std;

string food_security = "wm/concept/causal_factor/condition/food_security";
string inflation = "wm/concept/causal_factor/economic_and_commerce/economic_activity/market/inflation";
string tension = "wm/concept/causal_factor/condition/tension";
string displacement = "wm/concept/indicator_and_reported_property/conflict/population_displacement";
string crop_production = "wm/concept/indicator_and_reported_property/agriculture/Crop_Production";


void test_simple_path_construction() {
  AnalysisGraph G = AnalysisGraph
          ::from_indra_statements_json_file(
              "../tests/data/indra_statements_format.json");

  G.add_node("c0");
  G.add_node("c1");
  G.add_node("c2");

  cout << "Nodes of the graph:\n";
  G.print_nodes();

  G.add_edge({{"", 1, "c0"}, {"", 1, "c1"}});
  G.add_edge({{"", 1, "c1"}, {"", 1, "c2"}});
  G.add_edge({{"", 1, "c2"}, {"", 1, "c3"}});
  // Creates a loop 1->2->3->1
  G.add_edge({{"", 1, "c3"}, {"", 1, "c1"}});

  cout << "Edges of the graph:\n";
  G.print_edges();

  G.find_all_paths();
  G.print_all_paths();

  G.print_cells_affected_by_beta(0, 1);
  G.print_cells_affected_by_beta(1, 2);

  AnalysisGraph G2 = AnalysisGraph::from_indra_statements_json_file(
      "../tests/data/indra_statements_format.json");
}


void test_inference() {
  vector<CausalFragment> causal_fragments = {
      {{"small", 1, tension}, {"large", -1, food_security}}};

  cout << "\n\nCreating CAG\n";
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);

  G.print_nodes();

  cout << "\nName to vertex ID map entries\n";
  G.print_name_to_vertex();
}

void test_remove_node() {
  cout << "\n\n=====================================\ntest_remove_node\n\n";
  vector<CausalFragment> causal_fragments = {
      {{"small", 1, tension}, {"large", -1, food_security}}};

  cout << "\nCreating CAG\n";
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);

  G.find_all_paths();
  G.print_nodes();

  cout << "\nRemoving an invalid concept\n";
  try {
    G.remove_node("invalid");
  }
  catch (out_of_range& e) {
    cout << "Tried to remove an invalid node\n" << e.what();
  };

  cout << "\nRemoving a valid concept\n";
  G.remove_node(tension);
  G.print_nodes();
}

void test_remove_nodes() {
  cout << "\n\n=====================================\ntest_remove_nodes\n\n";
  vector<CausalFragment> causal_fragments = {
      {{"small", 1, tension}, {"large", -1, food_security}}};

  cout << "\nCreating CAG\n";
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);

  G.find_all_paths();
  G.print_nodes();

  cout << "\nName to vertex ID map entries\n";
  G.print_name_to_vertex();

  G.print_all_paths();

  cout << "\nRemoving a several concepts, some valid, some invalid\n";
  G.remove_nodes({"invalid1", tension, "invalid2"});

  G.print_nodes();

  cout << "\nName to vertex ID map entries\n";
  G.print_name_to_vertex();
  G.print_all_paths();
}

void test_remove_edge() {
  cout << "\n\n=====================================\ntest_remove_edge\n\n";
  vector<CausalFragment> causal_fragments = {
      {{"small", 1, tension}, {"large", -1, food_security}}};

  cout << "\nCreating CAG\n";
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);

  G.find_all_paths();
  G.print_nodes();
  G.print_all_paths();

  cout << "\nRemoving edge - invalid source\n";
  try {
    G.remove_edge("invalid", food_security);
  }
  catch (out_of_range& e) {
    cout << "Tried to remove an edge invalid source\n" << e.what();
  };

  cout << "\nRemoving edge - invalid target\n";
  try {
    G.remove_edge(tension, "invalid");
  }
  catch (out_of_range& e) {
    cout << "Tried to remove an edge invalid target\n" << e.what();
  };

  cout << "\nRemoving edge - source and target inverted target\n";
  G.remove_edge(food_security, tension);
  G.print_nodes();
  G.print_edges();
  G.print_all_paths();

  cout << "\nRemoving edge - correct\n";
  G.remove_edge(tension, food_security);
  G.print_nodes();
  G.print_edges();
  G.print_all_paths();
  G.to_png();
}

void test_remove_edges() {
  cout << "\n\n=====================================\ntest_remove_edges\n\n";
  vector<CausalFragment> causal_fragments = {
      {{"small", 1, tension}, {"large", -1, food_security}}};

  cout << "\nCreating CAG\n";
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);
  G.to_png("test_CAG.png");
  G.find_all_paths();
  G.print_nodes();
  cout << "\nName to vertex ID map entries\n";
  G.print_name_to_vertex();
  G.print_all_paths();

  vector<pair<string, string>> edges_to_remove =
      {
          {"invalid_src_1", food_security},
          {"invalid_src_2", food_security},
          {tension, "invalid_tgt1"},
          {tension, "invalid_tgt2"},
          {"invalid_src_2", "invalid_tgt_2"},
          {"invalid_src_3", "invalid_tgt3"},
          {food_security, tension},
          {tension, food_security},
      };

  cout << "\nRemoving edges\n";
  G.remove_edges(edges_to_remove);
  G.print_nodes();
  cout << "\nName to vertex ID map entries\n";
  G.print_name_to_vertex();
  G.print_edges();
  G.print_all_paths();
}


void test_subgraph() {

  cout << "\n\n=====================================\ntest_subgraph\n\n";

  vector<CausalFragment> causal_fragments = {  // Center node is n4
      {{"small", 1, "n0"}, {"large", -1, "n1"}},
      {{"small", 1, "n1"}, {"large", -1, "n2"}},
      {{"small", 1, "n2"}, {"large", -1, "n3"}},
      {{"small", 1, "n3"}, {"large", -1, "n4"}},
      {{"small", 1, "n4"}, {"large", -1, "n5"}},
      {{"small", 1, "n5"}, {"large", -1, "n6"}},
      {{"small", 1, "n6"}, {"large", -1, "n7"}},
      {{"small", 1, "n7"}, {"large", -1, "n8"}},
      {{"small", 1, "n0"}, {"large", -1, "n9"}},
      {{"small", 1, "n9"}, {"large", -1, "n2"}},
      {{"small", 1, "n2"}, {"large", -1, "n10"}},
      {{"small", 1, "n10"}, {"large", -1, "n4"}},
      {{"small", 1, "n4"}, {"large", -1, "n11"}},
      {{"small", 1, "n11"}, {"large", -1, "n6"}},
      {{"small", 1, "n6"}, {"large", -1, "n12"}},
      {{"small", 1, "n12"}, {"large", -1, "n8"}},
      {{"small", 1, "n13"}, {"large", -1, "n14"}},
      {{"small", 1, "n14"}, {"large", -1, "n4"}},
      {{"small", 1, "n4"}, {"large", -1, "n15"}},
      {{"small", 1, "n15"}, {"large", -1, "n16"}},
      {{"small", 1, "n5"}, {"large", -1, "n3"}},  // Creates a loop
  };

  cout << "\nCreating CAG\n";
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);
  G.find_all_paths();
  G.print_nodes();
  cout << "\nName to vertex ID map entries\n";
  G.print_name_to_vertex();

  int hops = 2;
  string node = "n4";
  cout << "\nSubgraph of " << hops << " hops beginning at node " << node
       << " graph\n";

  AnalysisGraph G_sub;

  try {
    G_sub = G.get_subgraph_for_concept(node, false, hops);
  }
  catch (out_of_range& e) {
    cout << "Concept " << node << " is not in the CAG!\n";
    return;
  }

  cout << "\n\nTwo Graphs\n";
  cout << "The original\n";
  G.print_nodes();
  G.print_name_to_vertex();

  cout << "The subgraph\n";
  G_sub.print_nodes();
  G_sub.print_name_to_vertex();

  cout << "\nSubgraph of " << hops << " hops ending at node " << node
       << " graph\n";
  G_sub = G.get_subgraph_for_concept(node, true, hops);

  cout << "\n\nTwo Graphs\n";
  cout << "The original\n";
  G.print_nodes();
  G.print_name_to_vertex();

  cout << "\nSubgraph of " << hops << " hops beginning at node " << node
       << " graph\n";

  // TODO; Segfault
  G.get_subgraph_for_concept(node, false, hops);
}


void test_subgraph_between() {

  cout << "\n\n=====================================\ntest_subgraph_between\n\n";

  vector<CausalFragment> causal_fragments = {  // Center node is n4
      {{"small", 1, "n0"}, {"large", -1, "n1"}},
      {{"small", 1, "n1"}, {"large", -1, "n2"}},
      {{"small", 1, "n2"}, {"large", -1, "n3"}},
      {{"small", 1, "n3"}, {"large", -1, "n4"}},
      {{"small", 1, "n0"}, {"large", -1, "n5"}},
      {{"small", 1, "n5"}, {"large", -1, "n6"}},
      {{"small", 1, "n6"}, {"large", -1, "n4"}},
      {{"small", 1, "n0"}, {"large", -1, "n7"}},
      {{"small", 1, "n7"}, {"large", -1, "n4"}},
      {{"small", 1, "n0"}, {"large", -1, "n4"}},
      {{"small", 1, "n0"}, {"large", -1, "n8"}},
      {{"small", 1, "n8"}, {"large", -1, "n9"}},
      {{"small", 1, "n10"}, {"large", -1, "n0"}},
      {{"small", 1, "n4"}, {"large", -1, "n12"}},
      {{"small", 1, "n12"}, {"large", -1, "n13"}},
      {{"small", 1, "n13"}, {"large", -1, "n4"}},
  };

  cout << "\nCreating CAG\n";
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);
  G.find_all_paths();
  G.print_nodes();
  cout << "\nName to vertex ID map entries\n";
  G.print_name_to_vertex();

  int cutoff = 3;
  string src = "n0";
  string tgt = "n4";

  cout << "\nSubgraph with in between hops less than or equal " << cutoff
       << " between source node " << src
       << "  and target node " << tgt << endl;

  AnalysisGraph G_sub;

  try {
    G_sub = G.get_subgraph_for_concept_pair(src, tgt, cutoff);
  }
  catch (out_of_range& e) {
    cout << "Incorrect source or target concept!\n";
    return;
  }

  cout << "\n\nTwo Graphs\n";
  cout << "The original\n";
  G.print_nodes();
  G.print_name_to_vertex();

  cout << "The subgraph\n";
  G_sub.print_nodes();
  G_sub.print_name_to_vertex();
}


void test_prune() {

  cout << "\n\n=====================================\ntest_prune\n\n";

  vector<CausalFragment> causal_fragments = {
      // Center node is n4
      {{"small", 1, "n0"}, {"large", -1, "n1"}},
      {{"small", 1, "n0"}, {"large", -1, "n2"}},
      {{"small", 1, "n0"}, {"large", -1, "n3"}},
      {{"small", 1, "n2"}, {"large", -1, "n1"}},
      {{"small", 1, "n3"}, {"large", -1, "n4"}},
      {{"small", 1, "n4"}, {"large", -1, "n1"}},
  };

  cout << "\nCreating CAG\n";
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);
  G.find_all_paths();

  cout << "\nBefore pruning\n";
  G.print_all_paths();

  int cutoff = 2;

  G.prune(cutoff);

  cout << "\nAfter pruning\n";
  G.print_all_paths();
}


void test_merge() {

  cout << "\n\n=====================================\ntest_merge\n\n";

  vector<CausalFragment> causal_fragments = {
      {{"small", 1, tension}, {"large", -1, food_security}},
      {{"small", 1, displacement}, {"small", 1, tension}},
      {{"small", 1, displacement}, {"large", -1, food_security}},
      {{"small", 1, tension}, {"small", 1, crop_production}},
      {{"large", -1, food_security}, {"small", 1, crop_production}},
      {
          {"small", 1, "UN/events/human/economic_crisis"},
              {"small", 1, tension},
      },
      {
          {"small", 1, "UN/events/weather/precipitation"},
              {"large", -1, food_security},
      },
      {{"large", -1, food_security}, {"small", 1, inflation}},
  };

  cout << "\nCreating CAG\n";
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);
  G.find_all_paths();

  cout << "\nBefore merging\n";
  G.print_all_paths();
  G.print_nodes();

  cout << "\nAfter merging\n";
  G.merge_nodes(food_security, tension);

  G.print_all_paths();
  G.print_nodes();
}


void test_debug() {

  cout << "\n\n=====================================\ntest_debug\n\n";

  vector<CausalFragment> causal_fragments = {
      // Center node is n4
      {{"small", 1, "n0"}, {"large", -1, "n1"}},
      {{"small", 1, "n0"}, {"large", -1, "n2"}},
      {{"small", 1, "n0"}, {"large", -1, "n3"}},
      {{"small", 1, "n2"}, {"large", -1, "n1"}},
      {{"small", 1, "n3"}, {"large", -1, "n4"}},
      {{"small", 1, "n4"}, {"large", -1, "n1"}},
  };

  vector<CausalFragment> causal_fragments2 = {
      // Center node is n4
      {{"small", 1, "n0"}, {"large", -1, "n1"}},
      {{"small", 1, "n1"}, {"large", -1, "n2"}},
      {{"small", 1, "n2"}, {"large", -1, "n3"}},
      {{"small", 1, "n3"}, {"large", -1, "n4"}},
      {{"small", 1, "n4"}, {"large", -1, "n5"}},
      {{"small", 1, "n5"}, {"large", -1, "n6"}},
      {{"small", 1, "n6"}, {"large", -1, "n7"}},
      {{"small", 1, "n7"}, {"large", -1, "n8"}},
      {{"small", 1, "n0"}, {"large", -1, "n9"}},
      {{"small", 1, "n9"}, {"large", -1, "n2"}},
      {{"small", 1, "n2"}, {"large", -1, "n10"}},
      {{"small", 1, "n10"}, {"large", -1, "n4"}},
      {{"small", 1, "n4"}, {"large", -1, "n11"}},
      {{"small", 1, "n11"}, {"large", -1, "n6"}},
      {{"small", 1, "n6"}, {"large", -1, "n12"}},
      {{"small", 1, "n12"}, {"large", -1, "n8"}},
      {{"small", 1, "n13"}, {"large", -1, "n14"}},
      {{"small", 1, "n14"}, {"large", -1, "n4"}},
      {{"small", 1, "n4"}, {"large", -1, "n15"}},
      {{"small", 1, "n15"}, {"large", -1, "n16"}},
      {{"small", 1, "n5"}, {"large", -1, "n3"}}, // Creates a loop
  };

  vector<CausalFragment> causal_fragments3 = {
      // Center node is n4
      {{"small", 1, "n0"}, {"large", -1, "n1"}},
      {{"small", 1, "n1"}, {"large", -1, "n2"}},
      {{"small", 1, "n2"}, {"large", -1, "n3"}},
      {{"small", 1, "n3"}, {"large", -1, "n4"}},
      {{"small", 1, "n4"}, {"large", -1, "n5"}},
      {{"small", 1, "n5"}, {"large", -1, "n6"}},
      {{"small", 1, "n6"}, {"large", -1, "n7"}},
      {{"small", 1, "n7"}, {"large", -1, "n8"}},
      {{"small", 1, "n0"}, {"large", -1, "n3"}}};

  cout << "\nCreating CAG\n";
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);
  G.find_all_paths();

  cout << "\nBefore pruning\n";
  G.print_all_paths();

  int hops = 3;
  string node = "n0";
  cout << "\nSubgraph of " << hops << " hops beginning at node " << node
       << " graph\n";

  AnalysisGraph G_sub;

  try {
    G_sub = G.get_subgraph_for_concept(node, false, hops);
  }
  catch (out_of_range& e) {
    cout << "Concept " << node << " is not in the CAG!\n";
    return;
  }

  G_sub.find_all_paths();
  G_sub.print_nodes();
}



int main(int argc, char* argv[]) {
  using namespace std;
  using namespace boost::program_options;

  vector<CausalFragment> causal_fragments2 = {  
        {{"small", 1, "n0"}, {"large", -1, "n1"}},
        {{"small", 1, "n1"}, {"large", -1, "n2"}},
  };

  cout << "Creating model\n";
  AnalysisGraph G = AnalysisGraph::from_causemos_json_file(
    "../tests/data/delphi/create_model_rain--temperature--yield.json", 0);

  cout << "\nOriginal\n" << G.generate_create_model_response() << "\n-----------------------\n" << endl;

  G = AnalysisGraph::deserialize_from_json_string(G.serialize_to_json_string(false), false);

  cout << "\nAfter serializing and deserializing\n" << G.generate_create_model_response() << "\n-----------------------\n" << endl;

  G.set_n_kde_kernels(100);
  G.run_train_model(10, 10);

  cout << "\nAfter training\n" << G.generate_create_model_response() << "\n-----------------------\n" << endl;

  G = AnalysisGraph::deserialize_from_json_string(G.serialize_to_json_string(false), false);

  cout << "\nAfter training and serializing and deserializing\n" << G.generate_create_model_response() << "\n-----------------------\n" << endl;

  G.freeze_edge_weight(
      "wm/concept/environment/meteorology/precipitation",
      "wm/concept/agriculture/crop_produce",
      0.25, 1);  // 0.392699 radians

  cout << "\nAfter freezing edge\n" << G.generate_create_model_response() << "\n-----------------------\n" << endl;

  G = AnalysisGraph::deserialize_from_json_string(G.serialize_to_json_string(false), false);

  cout << "\nAfter freezing edge and serializing and deserializing\n" << G.generate_create_model_response() << "\n-----------------------\n" << endl;

  G.set_n_kde_kernels(100);
  G.run_train_model(10, 10);

  cout << "\nAfter freezing edge and serializing and deserializing and training\n" << G.generate_create_model_response() << "\n-----------------------\n" << endl;

  G = AnalysisGraph::deserialize_from_json_string(G.serialize_to_json_string(false), false);

  cout << "\nAfter all\n" << G.generate_create_model_response() << "\n-----------------------\n" << endl;

  return 0;

  G.freeze_edge_weight(
      "wm/concept/environment/meteorology/precipitation",
      "wm/concept/environment/meteorology/temperature",
      0.75, 1);

  cout << G.generate_create_model_response() << "\n-----------------------\n" << endl;

  return 0;

  string initial_model = G.serialize_to_json_string(false);

  AnalysisGraph G2 = AnalysisGraph::deserialize_from_json_string(initial_model, false);
  unsigned short status = G2.freeze_edge_weight(
                            "wm/concept/environment/meteorology/precipitation",
                            "wm/concept/agriculture/crop_produce",
                            0.25, 1);  // 0.392699 radians
  if (status == 0) {
        G2.print_edges();
  }
  else {
      cout << "Error: " << status << endl;
  }

  G2.set_n_kde_kernels(100);
  G2.run_train_model(10, 10);
  cout << nlohmann::json::parse(G2.generate_create_model_response()).dump(2);

  FormattedProjectionResult proj;
  string frozen = G2.serialize_to_json_string(false);

  AnalysisGraph G3 = AnalysisGraph::deserialize_from_json_string(frozen, false);
  G3.print_edges();
  G3.set_n_kde_kernels(100);
  G3.run_train_model(10, 10);
  cout << nlohmann::json::parse(G2.generate_create_model_response()).dump(2);

  proj = G3.run_causemos_projection_experiment_from_json_file(
      "../tests/data/delphi/experiments_rain--temperature--yield.json");
  return(0);

  test_simple_path_construction();
  test_inference();
  test_remove_node();
  test_remove_nodes();
  test_remove_edge();
  test_remove_edges();
  test_subgraph();
  test_subgraph_between();
  test_prune();
  test_merge();
  test_debug();

    string result = G.serialize_to_json_string(false);
    G = AnalysisGraph::deserialize_from_json_string(result, false);
    proj = G.run_causemos_projection_experiment_from_json_file(
        "../tests/data/delphi/experiments_projection_input_2.json");
    string result2 = G.serialize_to_json_string(false);

    if (result.compare(result2) == 0){
      cout << "\nSame\n";
    } else {
      cout << "\nDifferent\n";
    }

    G.train_model(2020, 1, 2020, 12);
    result = G.serialize_to_json_string(false);

    //for(int i = 0; i < 100; i++)
    {
      //std::cout << "Iteration: " << i << std::endl << "-----------------------" << std::endl;
      //auto G = AnalysisGraph::from_causal_fragments(causal_fragments2);
      auto G = AnalysisGraph();
      G.add_node("n0");
      G.add_node("n1");
      G.add_node("n2");
      //G.add_edge("n0", "n1");
      G.add_edge("n1", "n2");
      G.remove_node("n0");
      //G.set_indicator("n0", "test_ind", "test_source");
      //G.set_indicator("n0", "test_ind2", "test_source");
      //G.set_indicator("n10", "test_ind3", "test_source");
      //G.set_indicator("n4", "test_ind4", "test_source");
  
      G.find_all_paths();
      auto hops = 2;
      auto node = "n4";
      //unordered_set<string> nodes_to_remove = {"n14", "n13", "n10", "n9", "n12", "n8", "n7", "n2", "n1", "n0"};
      //std::cout << "In create_model.cpp: before get_subgraph_for_concept" << std::endl;
      //G.print_nodes();
      /*
      for(auto n: nodes_to_remove)
      {
          G.remove_node(n);
      }
      */
      //G.remove_node("n13");
      //G.print_nodes();
      //G.print_edges();
      //G.print_indicators();
      //auto G_sub = G.get_subgraph_for_concept(node, false, hops);
      //auto G_sub = G;
      //G_sub.print_nodes();
      //G_sub.print_edges();
      //G_sub.print_indicators();
      //G.print_nodes();
      //std::cout << "In create_model.cpp: after get_subgraph_for_concept" << std::endl;
    }
  }
