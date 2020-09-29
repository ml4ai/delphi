// Just a cpp with a main to quickly test methods implemented in sandbox.cpp

#include "AnalysisGraph.hpp"
#include <boost/program_options.hpp>
#include "dbg.h"
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace boost::program_options;

    ofstream json_orig;
    AnalysisGraph G = AnalysisGraph::from_causemos_json_file("../tests/data/delphi_create_model_payload.json");
    json_orig.open ("json_original_verbose.txt");
    string json_verbose = G.serialize_to_json_string();
    json_orig.close();
    json_orig.open ("json_original_compact.txt");
    string json_compact = G.serialize_to_json_string(false);
    json_orig.close();

    json_orig.open ("json_deserialized_verbose.txt");
    AnalysisGraph GD1 = AnalysisGraph::deserialize_from_json_file("json_original_verbose.txt"); 
    string json_verbose1 = GD1.serialize_to_json_string();
    json_orig << json_verbose1;
    json_orig.close();
    json_orig.open ("json_deserialized_compact.txt");
    AnalysisGraph GD2 = AnalysisGraph::deserialize_from_json_file("json_original_compact.txt"); 
    string json_compact1 = GD2.serialize_to_json_string(false);
    json_orig << json_compact1;
    json_orig.close();
    

    return(0);
}
