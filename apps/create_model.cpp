// Usage: ./create_model <path_to_JSON_file>

#include "AnalysisGraph.hpp"

using namespace std;

int main(int argc, char* argv[]){
  auto G = AnalysisGraph::from_json_file(argv[1]);
  G.get_subgraph_for_concept("UN/entities/human/food/food_security");
  G.to_png();
  return EXIT_SUCCESS;
}
