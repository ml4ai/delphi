// Just a cpp with a main to quickly test methods implemented in sandbox.cpp

#include "AnalysisGraph.hpp"
#include <boost/program_options.hpp>
#include "dbg.h"
#include <iostream>
#include <fstream>

// TODO: For debugging remove later.
using fmt::print;


int main(int argc, char* argv[]) {
    using namespace std;
    using namespace boost::program_options;
    using fmt::print;

    //AnalysisGraph G = AnalysisGraph::from_causemos_json_file("../tests/data/delphi_create_model_payload.json", 4);
    AnalysisGraph G = AnalysisGraph::from_causemos_json_file("../tests/data/delphi/causemos_create-model.json", 4);
    G.train_model(2020, 1, 2020, 12);

    // Serialize the model
    string json_compact = G.serialize_to_json_string(false);
    string json_verbose = G.serialize_to_json_string(true);

    // Deserialize the serialized model
    AnalysisGraph G_from_compact = AnalysisGraph::deserialize_from_json_string(json_compact, false); 

    // Serialize these models again.
    string json_compact_2 = G_from_compact.serialize_to_json_string(false);
    G_from_compact.train_model(2020, 1, 2020, 12);

    // Let's check whether they agree.
    // NOTE: It seems they do not agree. This needs to be fixed
    if (json_compact.compare(json_compact_2) == 0) {
        print("Serialization: compact success\n");
    } else {
        print("Serialization: compact failure\n");
    }

    ofstream fs;
    fs.open ("json_original_compact.txt");
    fs << json_compact;
    fs.close();
    fs.open ("json_deserialized_compact.txt");
    fs << json_compact_2;
    fs.close();

    
    /*// NOTE: This line generates an error.
    // If this cannot be debugged quickly we can keep it to later.
    print("before \n");
    AnalysisGraph G_from_verbose = AnalysisGraph::deserialize_from_json_string(json_verbose, true); 
    print("after \n");
    // Serialize these models again.
    string json_verbose_2 = G_from_verbose.serialize_to_json_string(true);

    if (json_verbose.compare(json_verbose_2) == 0) {
        print("Serialization: verbose success\n");
    } else {
        print("Serialization: verbose failure\n");
    }

    fs.open ("json_original_verbose.txt");
    fs << json_verbose;
    fs.close();
    fs.open ("json_deserialized_verbose.txt");
    fs << json_verbose_2;
    fs.close();
    */

    /*
    ofstream json_orig;
    json_orig.open ("json_original_verbose.txt");
    json_orig.close();
    json_orig.open ("json_original_compact.txt");
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
    */


//=======
    
    //print("before \n");
    //string json_verbose = G.serialize_to_json_string();
    //print("middle \n");
    //string json_compact = G.serialize_to_json_string(false);
    //print("after \n");

    //json_orig.open ("json_original_verbose.txt");
    //string json_verbose = G.serialize_to_json_string();
    //json_orig.close();
    //json_orig.open ("json_original_compact.txt");
    //string json_compact = G.serialize_to_json_string(false);
    //json_orig.close();

    //json_orig.open ("json_deserialized_verbose.txt");
    //AnalysisGraph GD1 = AnalysisGraph::deserialize_from_json_file("json_original_verbose.txt"); 
    //string json_verbose1 = GD1.serialize_to_json_string();
    //json_orig << json_verbose1;
    //json_orig.close();

    //json_orig.open ("json_deserialized_compact.txt");
    //AnalysisGraph GD2 = AnalysisGraph::deserialize_from_json_file("json_original_compact.txt"); 
    //string json_compact1 = GD2.serialize_to_json_string(false);
    //json_orig << json_compact1;
    //json_orig.close();
//>>>>>>> Stashed changes
    

    return(0);
}
