#include <iostream>
#include <served/served.hpp>
#include <nlohmann/json.hpp>
#include "AnalysisGraph.hpp"
#include <string>
#include <assert.h>
#include <sqlite3.h>
#include <range/v3/all.hpp>
#include "DatabaseHelper.hpp"
#include <boost/graph/graph_traits.hpp>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <thread> 



using namespace std;
using json = nlohmann::json;

/*
    =====================
    Delphi - CauseMos API
    =====================
*/


void testcase_Database_Create(Database* sqlite3DB)
{
    std::cout << "\n======================= Test testcase_Database_Create =======================\n" << std::endl;
    
    sqlite3DB->Database_Create();
}


void testcase_Database_Read_ColumnText(Database* sqlite3DB)
{
    std::cout << "\n======================= Test testcase_Database_Read_ColumnText =======================\n" << std::endl;
    
    string query = "SELECT adjective from gradableAdjectiveData;";
    //string query = "select Unit from indicator;";
    vector<string> matches = sqlite3DB->Database_Read_ColumnText(query);
    cout << "Read vector size: " << matches.size() << endl;
}


void testcase_Database_Insert(Database* sqlite3DB)
{
    std::cout << "\n======================= Test testcase_Database_Insert =======================\n" << std::endl;
    
    string query = "INSERT INTO concept_to_indicator_mapping ('Concept', 'Source', 'Indicator', 'Score') VALUES ('wm/concept/time/temporal/crop_season', 'delphi_db_inds_TEST', 'TEST', 0.49010614);";
    sqlite3DB->Database_Insert(query);
    
    query = "SELECT Source from concept_to_indicator_mapping WHERE Indicator = 'TEST';";
    vector<string> matches = sqlite3DB->Database_Read_ColumnText(query);
    cout << "Read vector size: " << matches.size() << endl;
    if(matches.size()) cout << "Read Source column value: " << matches[0] << endl;
}

void testcase_Database_Update(Database* sqlite3DB)
{
    std::cout << "\n======================= Test testcase_Database_Update =======================\n" << std::endl;
    
    sqlite3DB->Database_Update("concept_to_indicator_mapping", "Source", "delphi_db_inds_TEST_update_2", "Indicator", "TEST");
        void Database_Update(string table_name, string column_name, string value, string where_column_name, string where_value);

    string query = "SELECT Source from concept_to_indicator_mapping WHERE Indicator = 'TEST';";
    vector<string> matches = sqlite3DB->Database_Read_ColumnText(query);
    cout << "Read vector size: " << matches.size() << endl;
    if(matches.size()) cout << "Read Source column value: " << matches[0] << endl;
}


void testcase_delete(Database* sqlite3DB)
{
    std::cout << "\n======================= Test testcase_Experiment =======================\n" << std::endl;
    
 
    sqlite3DB->Database_Delete_Rows("delphimodel", "id", "XYZ");
    string curl_create_model = "curl -X POST \"http://localhost:8123/delphi/create-model\" -d @../tests/data/delphi/causemos_create-model.json --header \"Content-Type: application/json\"";  
    string curl_create_experiment = "curl -X POST \"http://localhost:8123/delphi/models/XYZ/experiments\" -d @../tests/data/delphi/causemos_experiments_projection_input.json --header \"Content-Type: application/json\"";

    system(curl_create_model.c_str());
    system(curl_create_experiment.c_str());
}




class Experiment{
public:

    /* a function to generate numpy's linspace */
    static vector<double> linspace(double start, double end, double num)
    {
        vector<double> linspaced;
    
        if (0 != num)
        {
            if (1 == num) 
            {
                linspaced.push_back(round(static_cast<double>(start)));
            }
            else
            {
                double delta = (end - start) / (num - 1);
    
                for (auto i = 0; i < (num - 1); ++i)
                {
                    linspaced.push_back(round(static_cast<double>(start + delta * i)));
                }
                // ensure that start and end are exactly the same as the input
                linspaced.push_back(round(static_cast<double>(end)));
            }
        }
        return linspaced;
    }
    
    
    
    
    static void runProjectionExperiment(Database* sqlite3DB, const served::request & request, string modelID, string experiment_id, AnalysisGraph G, bool trained){
        cout << "Start  runProjectionExperiment" << endl;
        auto request_body = nlohmann::json::parse(request.body());
    
    
        double startTime = request_body["experimentParam"]["startTime"];
        double endTime = request_body["experimentParam"]["endTime"];
        int numTimesteps = request_body["experimentParam"]["numTimesteps"];
    
        cout << "Before FormattedProjectionResult" << endl;
        FormattedProjectionResult causemos_experiment_result = G.run_causemos_projection_experiment(
            request.body() 
        );
        cout << "After FormattedProjectionResult" << endl;
    
        if(!trained){
            sqlite3DB->Database_InsertInto_delphimodel(modelID, G.serialize_to_json_string(false));
            cout << "rPE: After Database_InsertInto_delphimodel" << endl;
        }
    
        json result = sqlite3DB->Database_Read_causemosasyncexperimentresult(experiment_id);
        cout << "rPE: After Database_Read_causemosasyncexperimentresult" << endl;
    
        /* 
            A rudimentary test to see if the projection failed. We check whether
            the number time steps is equal to the number of elements in the first
            concept's time series.
        */
        vector<vector<vector<double>>> formattedProjectionTimestepSample;
        for (const auto & [ key, value ] : causemos_experiment_result) {
            formattedProjectionTimestepSample.push_back(value);
        }
        if(formattedProjectionTimestepSample[0].size() < numTimesteps)
            result["status"] = "failed";
        else{
            result["status"] = "completed";
    
            vector<double> timesteps_nparr = linspace(startTime, endTime, numTimesteps); 
    
            /* 
                The calculation of the 95% confidence interval about the median is
                taken from:
                https://www.ucl.ac.uk/child-health/short-courses-events/ \
                about-statistical-courses/research-methods-and-statistics/chapter-8-content-8
            */
            int n = G.get_res();
    
            int lower_rank = (int)((n - 1.96 * sqrt(n)) / 2);
            int upper_rank = (int)((2 + n + 1.96 * sqrt(n)) / 2);
    
            lower_rank = lower_rank < 0 ? 0 : lower_rank;
            upper_rank = upper_rank >= n ? n - 1 : upper_rank;
            unordered_map<string, vector<string>> res_data; 
            res_data["data"] = {};
            
            result["results"] = res_data;
    
            for (const auto & [ conceptname, timestamp_sample_matrix ] : causemos_experiment_result) {
                json data_dict;
                data_dict["concept"] = conceptname;
                data_dict["values"] = vector<unordered_map<string, double>>{};
                unordered_map<string, vector<double>> m { {"upper", vector<double>{}}, {"lower", vector<double>{}}};
                data_dict["confidenceInterval"] = m;
                for(int i = 0; i < timestamp_sample_matrix.size(); i++){
                    vector<double> time_step = timestamp_sample_matrix[i];
                    sort(time_step.begin(), time_step.end());
                    int l = time_step.size() / 2;
                    double median_value = 0;
                    if(time_step.size())
                        median_value = time_step.size() % 2? time_step[l]: (time_step[l] + time_step[l - 1]) / 2;
    
                    double lower_limit = time_step[lower_rank];
                    double upper_limit = time_step[upper_rank];
    
                    unordered_map<string, double> value_dict = {
                        {"timestamp", timesteps_nparr[i]},
                        {"value", median_value}
                    };
    
                    data_dict["values"].push_back(value_dict);
                    value_dict["value"] = lower_limit;
                    data_dict["confidenceInterval"]["lower"].push_back(value_dict);
    
                    value_dict["value"] = upper_limit;
                    data_dict["confidenceInterval"]["upper"].push_back(value_dict);
                }
                result["results"]["data"].push_back(data_dict); 
            }
        }
        cout << "rPE: After computing result" << endl;
    
        sqlite3DB->Database_InsertInto_causemosasyncexperimentresult(result["id"], result["status"], result["experimentType"], result["results"].dump());
        cout << "rPE: After Database_InsertInto_causemosasyncexperimentresult with result --- STOP Thread" << endl;
        cout << "End  runProjectionExperiment" << endl;
    }
    
    
    
    static void runExperiment(Database* sqlite3DB, const served::request & request, string modelID, string experiment_id){
        cout << "Start  runExperiment" << endl;
        cout << "process id : "  << getpid() << "  thread id : "<< this_thread::get_id() << endl; 
        auto request_body = nlohmann::json::parse(request.body());
        string experiment_type = request_body["experimentType"]; 
    
        json query_result = sqlite3DB->Database_Read_delphimodel(modelID);
        cout << "After  Database_Read_delphimodel " << query_result["id"]<< endl;
    
        if(query_result.empty()){ 
            // Model ID not in database. Should be an incorrect model ID
            query_result = sqlite3DB->Database_Read_causemosasyncexperimentresult(experiment_id);
            cout << "After  Database_Read_causemosasyncexperimentresult" << endl;
            cout << "Before  Database_InsertInto_causemosasyncexperimentresult" << query_result["id"] << endl;
            sqlite3DB->Database_InsertInto_causemosasyncexperimentresult(query_result["id"], "failed", query_result["experimentType"], query_result["results"]);
            cout << "After  Database_InsertInto_causemosasyncexperimentresult" << endl;
            return;
        }
    
        string model = query_result["model"];   
        bool trained = nlohmann::json::parse(model)["trained"];
    
        cout << "After  trained " << trained << endl;  
        AnalysisGraph G;
        G = G.deserialize_from_json_string(model, false);
        cout << "After  from_delphi_json_dict" << endl;
    
        if(experiment_type == "PROJECTION")
            runProjectionExperiment(sqlite3DB, request, modelID, experiment_id, G, trained); 
        else if(experiment_type == "GOAL_OPTIMIZATION")
            ;// Not yet implemented
        else if( experiment_type == "SENSITIVITY_ANALYSIS")
            ;// Not yet implemented
        else if( experiment_type == "MODEL_VALIDATION")
            ;// Not yet implemented
        else if( experiment_type == "BACKCASTING")
            ;// Not yet implemented
        else
            ;// Unknown experiment type
        cout << "End  runExperiment" << endl;
    }

};





int main(int argc, const char *argv[])
{
    Database* sqlite3DB = new Database();
    Experiment* experiment = new Experiment();


    //================= Testsuite =============================================
    // Run Tests:
    //testcase_Database_Create(sqlite3DB); 
    
    // working
    //testcase_Database_Read_ColumnText(sqlite3DB); 
    //testcase_Database_Insert(sqlite3DB); 
    //testcase_Database_Update(sqlite3DB); 
    // CLEANUP: drop test rows from delphi.db
    //testcase_delete(sqlite3DB);

    //======================App=================================================    
    served::multiplexer mux;


    /* Create a new Delphi model. */
    mux.handle("/delphi/create-model")
        .post([&sqlite3DB](served::response & response, const served::request & req) {
            nlohmann::json json_data = nlohmann::json::parse(req.body());
            /*
             If neither "CI" or "DELPHI_N_SAMPLES" is set, we default to a
             sampling resolution of 1000.
   
             TODO - we might want to set the default sampling resolution with some
             kind of heuristic, based on the number of nodes and edges. - Adarsh
            */
            size_t res = 1000;
            if(getenv("CI")) 
                // When running in a continuous integration run, we set the sampling
                // resolution to be small to prevent timeouts.
                res = 5;
            else if (getenv("DELPHI_N_SAMPLES")) {
                // We also enable setting the sampling resolution through the
                // environment variable "DELPHI_N_SAMPLES", for development and testing
                // purposes.
                res = (size_t)stoul(getenv("DELPHI_N_SAMPLES"));
                cout << getenv("DELPHI_N_SAMPLES") << endl;
            }

            AnalysisGraph G;
            G.set_res(res);
            G.from_causemos_json_dict(json_data);
            cout << "After  from_causemos_json_dict" << endl;

            sqlite3DB->Database_InsertInto_delphimodel(json_data["id"], G.serialize_to_json_string(false));
            cout << "After  Database_InsertInto_delphimodel" << endl;
            response << json_data.dump();
            cout << "END  createmodel" << endl;
            return json_data.dump();

            //string strresult = json_data.dump();
            //response << strresult;
            //cout << "END  createmodel" << endl;
            //return strresult;
        });




    /* 
         """ Fetch experiment results"""
         Get asynchronous response while the experiment is being trained from
         causemosasyncexperimentresult table

         NOTE: I saw some weird behavior when we request results for an invalid
         experiment ID just after running an experiment. The trained model seemed
         to be not saved to the database. The model got re-trained from scratch
         on a subsequent experiment after the initial experiment and the invalid
         experiment result request. When I added a sleep between the initial
         create experiment and the invalid result request this re-training did not
         occur. 
    */    
    mux.handle("/delphi/models/{modelID}/experiments/{experimentID}")
        .get([&sqlite3DB](served::response & res, const served::request & req) {
            json result = sqlite3DB->Database_Read_causemosasyncexperimentresult(req.params["experimentID"]);
            cout << result["experimentType"] << result["status"]  <<  endl;
            if(result.empty()){
                // experimentID not in database. Should be an incorrect experimentID
                result["experimentType"] = "UNKNOWN";
                result["status"] = "invalid experiment id";
                result["results"] = "";
            } 
            result["modelId"] = req.params["modelID"];
            result["experimentId"] = req.params["experimentID"];

            res << result.dump();

            //string strresult = result.dump();
            //res << strresult;
            //return strresult;
        });




    /*
        """ createCausemosExperiment """
        Create causemos asynchronous experiment by creating
        a new thread to run the projection and returning 
        served's REST response immediately.
    */    
    mux.handle("/delphi/models/{modelID}/experiments")
        .post([&sqlite3DB](served::response & res, const served::request & req) {
            auto request_body = nlohmann::json::parse(req.body());
            string modelID = req.params["modelID"];

            string experiment_type = request_body["experimentType"]; 
            boost::uuids::uuid uuid = boost::uuids::random_generator()(); 
            string experiment_id = to_string(uuid);

            cout << "Before Insert" << endl;
            sqlite3DB->Database_InsertInto_causemosasyncexperimentresult(experiment_id, "in progress", experiment_type, "");
            cout << "After Insert" << endl;

            cout << "process id : "  << getpid() << "  thread id : "<< this_thread::get_id() << endl; 
            thread executor_experiment (&Experiment::runExperiment, sqlite3DB, req, modelID, experiment_id);
            executor_experiment.detach();
            
            json ret_exp;
            ret_exp["experimentId"] = experiment_id;
            cout << "  REST handle  : ret_exp : " << ret_exp << endl;
            cout << "End  REST handle" << endl;
            res << ret_exp.dump();
            return ret_exp;


        });

    std::cout << "Run Delphi REST API with:" << std::endl;
    std::cout << "curl -X POST \"http://localhost:8123/delphi/create-model\" -d @../tests/data/delphi/causemos_create-model.json --header \"Content-Type: application/json\" " << std::endl;
    std::cout << "curl -X POST \"http://localhost:8123/delphi/models/XYZ/experiments\" -d @../tests/data/delphi/causemos_experiments_projection_input.json --header \"Content-Type: application/json\" " << std::endl;
    std::cout << "curl \"http://localhost:8123/delphi/models/XYZ/experiments/d93b18a7-e2a3-4023-9f2f-06652b4bba66\" " << std::endl;

    served::net::server server("127.0.0.1", "8123", mux);
    server.run(10);

    return (EXIT_SUCCESS);
}


