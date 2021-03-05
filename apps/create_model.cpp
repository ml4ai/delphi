// Usage: ./create_model <path_to_JSON_file>

#include "AnalysisGraph.hpp"
#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
  using namespace std;
  using namespace boost::program_options;

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

  vector<CausalFragment> causal_fragments2 = {  
        {{"small", 1, "n0"}, {"large", -1, "n1"}},
        {{"small", 1, "n1"}, {"large", -1, "n2"}},
  };


    AnalysisGraph G = AnalysisGraph::from_causemos_json_file(
      "../tests/data/delphi/create_model_input_2.json", 4);
    FormattedProjectionResult proj = G.run_causemos_projection_experiment_from_json_file(
        "../tests/data/delphi/experiments_projection_input_2.json",
        10,
        4);


  return(0);
    string result = G.serialize_to_json_string(false);
    G = AnalysisGraph::deserialize_from_json_string(result, false);
    proj = G.run_causemos_projection_experiment_from_json_file(
        "../tests/data/delphi/experiments_projection_input_2.json",
        10,
        4);
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
