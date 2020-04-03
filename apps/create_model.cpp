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



auto G = AnalysisGraph::from_causal_fragments(causal_fragments);
  
    G.find_all_paths();
    auto hops = 2;
    auto node = "n4";
    std::cout << "In create_model.cpp: before get_subgraph_for_concept" << std::endl;
    auto G_sub = G.get_subgraph_for_concept(node, false, hops);
    std::cout << "In create_model.cpp: after get_subgraph_for_concept" << std::endl;
  }
