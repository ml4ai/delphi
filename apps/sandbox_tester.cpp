// Just a cpp with a main to quickly test methods implemented in sandbox.cpp

#include "AnalysisGraph.hpp"
#include <boost/program_options.hpp>
#include "dbg.h"

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace boost::program_options;

    AnalysisGraph G = AnalysisGraph::from_causemos_json_file("../tests/data/delphi_create_model_payload.json");

    string json_verbose = G.serialize_to_json_string();
    string json_compact = G.serialize_to_json_string(false);


    return(0);
}
