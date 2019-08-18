// Usage: ./create_model <path_to_JSON_file>

#include "AnalysisGraph.hpp"

using namespace std;

int main(int argc, char* argv[]){
  auto G = AnalysisGraph::from_json_file(argv[1],0.8,0.5);
  G.get_subgraph_for_concept("UN/events/human/human_migration",2,true);
  G.prune(2);
  G.to_png();
  return EXIT_SUCCESS;
}
