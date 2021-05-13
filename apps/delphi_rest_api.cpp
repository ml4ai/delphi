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
        auto request_body = nlohmann::json::parse(request.body());
    
    
        double startTime = request_body["experimentParam"]["startTime"];
        double endTime = request_body["experimentParam"]["endTime"];
        int numTimesteps = request_body["experimentParam"]["numTimesteps"];
    
        FormattedProjectionResult causemos_experiment_result = G.run_causemos_projection_experiment_from_json_string(
            request.body()
        );
    
        if(!trained){
            sqlite3DB->insert_into_delphimodel(modelID, G.serialize_to_json_string(false));
        }
    
        json result = sqlite3DB->select_causemosasyncexperimentresult_row(experiment_id);
    
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
    
        sqlite3DB->insert_into_causemosasyncexperimentresult(result["id"], result["status"], result["experimentType"], result["results"].dump());
    }
    
    
    
    static void runExperiment(Database* sqlite3DB, const served::request & request, string modelID, string experiment_id){
        auto request_body = nlohmann::json::parse(request.body());
        string experiment_type = request_body["experimentType"]; 
    
        json query_result = sqlite3DB->select_delphimodel_row(modelID);
    
        if(query_result.empty()){ 
            // Model ID not in database. Should be an incorrect model ID
            query_result = sqlite3DB->select_causemosasyncexperimentresult_row(experiment_id);
            sqlite3DB->insert_into_causemosasyncexperimentresult(query_result["id"], "failed", query_result["experimentType"], query_result["results"]);
            return;
        }
    
        string model = query_result["model"];   
        bool trained = nlohmann::json::parse(model)["trained"];
    
        AnalysisGraph G;
        G = G.deserialize_from_json_string(model, false);
    
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
    }


    static void train_model(Database* sqlite3DB, AnalysisGraph G, string modelID, int sampling_resolution, int burn){
        G.run_train_model(sampling_resolution, burn, InitialBeta::ZERO, InitialDerivative::DERI_ZERO);
        sqlite3DB->insert_into_delphimodel(modelID, G.serialize_to_json_string(false));
    }

};





int main(int argc, const char *argv[])
{
    Database* sqlite3DB = new Database();
    Experiment* experiment = new Experiment();
    
    served::multiplexer mux;



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
            json result = sqlite3DB->select_causemosasyncexperimentresult_row(req.params["experimentID"]);
            if(result.empty()){
                // experimentID not in database. Should be an incorrect experimentID
                result["experimentType"] = "UNKNOWN";
                result["status"] = "invalid experiment id";
                result["results"] = "";
            } 
            result["modelId"] = req.params["modelID"];
            result["experimentId"] = req.params["experimentID"];

            res << result.dump();
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

            json query_result = sqlite3DB->select_delphimodel_row(modelID);
            
            if(query_result.empty()){ 
                json ret_exp;
                ret_exp["experimentId"] =  "invalid model id";
                res << ret_exp.dump();
                return ret_exp;
            }

            string model = query_result["model"];
            bool trained = nlohmann::json::parse(model)["trained"];

            if(trained == false){
                json ret_exp;
                ret_exp["experimentId"] =  "model not trained";
                res << ret_exp.dump();
                return ret_exp;
            }

            string experiment_type = request_body["experimentType"]; 
            boost::uuids::uuid uuid = boost::uuids::random_generator()(); 
            string experiment_id = to_string(uuid);

            sqlite3DB->insert_into_causemosasyncexperimentresult(experiment_id, "in progress", experiment_type, "");

            try{
                thread executor_experiment (&Experiment::runExperiment, sqlite3DB, req, modelID, experiment_id);
                executor_experiment.detach();
            }catch(std::exception &e){
                cout << "Error: unable to start experiment process" << endl;

                json ret_exp;
                ret_exp["experimentId"] =  "server error: experiment";
                res << ret_exp.dump();
                return ret_exp;
            }
            
            json ret_exp;
            ret_exp["experimentId"] = experiment_id;
            res << ret_exp.dump();
            return ret_exp;

        });




    /*
        getModelStatus
    */
    mux.handle("/delphi/models/{modelID}")
        .get([&sqlite3DB](served::response & res, const served::request & req) {
            string modelID = req.params["modelID"];
            json query_result = sqlite3DB->select_delphimodel_row(modelID);
            
            if(query_result.empty()){ 
                json ret_exp;
                ret_exp["status"] =  "invalid model id";
                res << ret_exp.dump();
                return;
            }

            string model = query_result["model"];

            AnalysisGraph G;
            G = G.deserialize_from_json_string(model, false);
            auto response = nlohmann::json::parse(G.generate_create_model_response());

            res << response.dump();
        });



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
            size_t kde_kernels = 1000;
            int sampling_resolution = 1000, burn = 10000;
            if(getenv("CI")) {
                // When running in a continuous integration run, we set the sampling
                // resolution to be small to prevent timeouts.
                kde_kernels = 5;
                sampling_resolution = 5;
                burn = 5;
            }
            else if (getenv("DELPHI_N_SAMPLES")) {
                // We also enable setting the sampling resolution through the
                // environment variable "DELPHI_N_SAMPLES", for development and testing
                // purposes.
                kde_kernels = (size_t)stoul(getenv("DELPHI_N_SAMPLES"));
                sampling_resolution = 100;
                burn = 100;
            }

            // ------------------
            burn = 100;
            // ------------------

            AnalysisGraph G;
            G.set_res(kde_kernels);
            G.from_causemos_json_dict(json_data, 0, 0);

            sqlite3DB->insert_into_delphimodel(json_data["id"], G.serialize_to_json_string(false));

            auto response_json = nlohmann::json::parse(G.generate_create_model_response());

            try{
                thread executor_create_model (&Experiment::train_model, sqlite3DB, G, json_data["id"], sampling_resolution, burn);
                executor_create_model.detach();
            }catch(std::exception &e){
                cout << "Error: unable to start training process" << endl;

                json ret_exp;
                ret_exp["status"] =  "server error: training";
                response << ret_exp.dump();
                return ret_exp.dump();
            }


            //response << response_json.dump();
            //return response_json.dump();

            string strresult = response_json.dump();
            response << strresult;
            return strresult;
        });



    std::cout << "Run Delphi REST API with:" << std::endl;
    std::cout << "curl -X POST \"http://localhost:8123/delphi/create-model\" -d @../tests/data/delphi/create_model_input_2.json --header \"Content-Type: application/json\" " << std::endl;
    std::cout << "curl \"http://localhost:8123/delphi/models/XYZ\" " << std::endl;
    std::cout << "curl -X POST \"http://localhost:8123/delphi/models/XYZ/experiments\" -d @../tests/data/delphi/experiments_projection_input_2.json --header \"Content-Type: application/json\" " << std::endl;
    std::cout << "curl \"http://localhost:8123/delphi/models/XYZ/experiments/d93b18a7-e2a3-4023-9f2f-06652b4bba66\" " << std::endl;

    served::net::server server("127.0.0.1", "8123", mux);
    server.run(10);

    return (EXIT_SUCCESS);
}

