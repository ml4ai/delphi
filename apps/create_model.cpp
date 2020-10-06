#include "AnalysisGraph.hpp"
#include <boost/program_options.hpp>
#include "dbg.h"

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace boost::program_options;

    AnalysisGraph G = AnalysisGraph::from_causemos_json_file("../tests/data/delphi/causemos_create-model.json");
    G.print_training_range();

    string model_json = G.serialize_to_json_string(false);

    //cout << "\n\nReturned json\n\n" << model_json << endl;

    return(0);
}
