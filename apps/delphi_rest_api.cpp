#include "AnalysisGraph.hpp"
#include "DatabaseHelper.hpp"
#include <assert.h>
#include <ctime>
#include <boost/graph/graph_traits.hpp>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <iostream>
#include <nlohmann/json.hpp>
#include <range/v3/all.hpp>
#include <served/served.hpp>
#include <sqlite3.h>
#include <string>
#include <thread>

using namespace std;
using json = nlohmann::json;

/*
    =====================
    Delphi - CauseMos API
    =====================
*/


// find computational bottlenecks
class Profiler {
    public:

	// let the user know the class has been instantiated
	static void test(){
            cout << "Profiler running\n";
	}

        // Test how long it takes to get an arbitrary model created
        static int initTiming(
	    Database* sqlite3DB,
            const served::request& request
	) {
	    cout << "ProfileExperiment.initTiming\n";

	    // let the compiler figure out the datatype particulars
            auto request_body = nlohmann::json::parse(request.body());

	    // These are set, they are not the profile data
            double startTime = request_body["experimentParam"]["startTime"];
            double endTime = request_body["experimentParam"]["endTime"];

	    return endTime-startTime;
	}

	// watch resources
	~Profiler() {
            cout << "Profiler destructor\n";
	}

     private:
	int n_iterations = 0;  // should be set by input
};


class Experiment {
    public:
    /* a function to generate numpy's linspace */
    static vector<double> linspace(double start, double end, double num) {
        vector<double> linspaced;

        if (0 != num) {
            if (1 == num) {
                linspaced.push_back(round(static_cast<double>(start)));
            }
            else {
                double delta = (end - start) / (num - 1);

                for (auto i = 0; i < (num - 1); ++i) {
                    linspaced.push_back(
                        round(static_cast<double>(start + delta * i)));
                }
                // ensure that start and end are exactly the same as the input
                linspaced.push_back(round(static_cast<double>(end)));
            }
        }
        return linspaced;
    }

    static void runProjectionExperiment(Database* sqlite3DB,
                                        const served::request& request,
                                        string modelId,
                                        string experiment_id,
                                        AnalysisGraph G,
                                        bool trained) {
        auto request_body = nlohmann::json::parse(request.body());

        double startTime = request_body["experimentParam"]["startTime"];
        double endTime = request_body["experimentParam"]["endTime"];
        int numTimesteps = request_body["experimentParam"]["numTimesteps"];

        FormattedProjectionResult causemos_experiment_result =
            G.run_causemos_projection_experiment_from_json_string(
                request.body());

        if (!trained) {
            sqlite3DB->insert_into_delphimodel(
                modelId, G.serialize_to_json_string(false));
        }

        json result =
            sqlite3DB->select_causemosasyncexperimentresult_row(experiment_id);

        /*
            A rudimentary test to see if the projection failed. We check whether
            the number time steps is equal to the number of elements in the
           first concept's time series.
        */
        vector<vector<vector<double>>> formattedProjectionTimestepSample;
        for (const auto& [key, value] : causemos_experiment_result) {
            formattedProjectionTimestepSample.push_back(value);
        }
        if (formattedProjectionTimestepSample[0].size() < numTimesteps)
            result["status"] = "failed";
        else {
            result["status"] = "completed";

            vector<double> timesteps_nparr =
                linspace(startTime, endTime, numTimesteps);

            /*
                The calculation of the 95% confidence interval about the median
               is taken from:
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

            for (const auto& [conceptname, timestamp_sample_matrix] :
                 causemos_experiment_result) {
                json data_dict;
                data_dict["concept"] = conceptname;
                data_dict["values"] = vector<unordered_map<string, double>>{};
                unordered_map<string, vector<double>> m{
                    {"upper", vector<double>{}}, {"lower", vector<double>{}}};
                data_dict["confidenceInterval"] = m;
                for (int i = 0; i < timestamp_sample_matrix.size(); i++) {
                    vector<double> time_step = timestamp_sample_matrix[i];
                    sort(time_step.begin(), time_step.end());
                    int l = time_step.size() / 2;
                    double median_value = 0;
                    if (time_step.size())
                        median_value =
                            time_step.size() % 2
                                ? time_step[l]
                                : (time_step[l] + time_step[l - 1]) / 2;

                    double lower_limit = time_step[lower_rank];
                    double upper_limit = time_step[upper_rank];

                    unordered_map<string, double> value_dict = {
                        {"timestamp", timesteps_nparr[i]},
                        {"value", median_value}};

                    data_dict["values"].push_back(value_dict);
                    value_dict["value"] = lower_limit;
                    data_dict["confidenceInterval"]["lower"].push_back(
                        value_dict);

                    value_dict["value"] = upper_limit;
                    data_dict["confidenceInterval"]["upper"].push_back(
                        value_dict);
                }
                result["results"]["data"].push_back(data_dict);
            }
        }

        sqlite3DB->insert_into_causemosasyncexperimentresult(
            result["id"],
            result["status"],
            result["experimentType"],
            result["results"].dump());
    }

    static void runExperiment(Database* sqlite3DB,
                              const served::request& request,
                              string modelId,
                              string experiment_id) {
        auto request_body = nlohmann::json::parse(request.body());
        string experiment_type = request_body["experimentType"];

        json query_result = sqlite3DB->select_delphimodel_row(modelId);

        if (query_result.empty()) {
            // Model ID not in database. Should be an incorrect model ID
            query_result = sqlite3DB->select_causemosasyncexperimentresult_row(
                experiment_id);
            sqlite3DB->insert_into_causemosasyncexperimentresult(
                query_result["id"],
                "failed",
                query_result["experimentType"],
                query_result["results"]);
            return;
        }

        string model = query_result["model"];
        bool trained = nlohmann::json::parse(model)["trained"];

        AnalysisGraph G;
        G = G.deserialize_from_json_string(model, false);

        if (experiment_type == "PROJECTION")
            runProjectionExperiment(
                sqlite3DB, request, modelId, experiment_id, G, trained);
        else if (experiment_type == "GOAL_OPTIMIZATION")
            ; // Not yet implemented
        else if (experiment_type == "SENSITIVITY_ANALYSIS")
            ; // Not yet implemented
        else if (experiment_type == "MODEL_VALIDATION")
            ; // Not yet implemented
        else if (experiment_type == "BACKCASTING")
            ; // Not yet implemented
        else
            ; // Unknown experiment type
    }

    static void train_model(Database* sqlite3DB,
                            AnalysisGraph G,
                            string modelId,
                            int sampling_resolution,
                            int burn) {
        G.run_train_model(sampling_resolution,
                          burn,
                          InitialBeta::ZERO,
                          InitialDerivative::DERI_ZERO);
        sqlite3DB->insert_into_delphimodel(modelId,
                                           G.serialize_to_json_string(false));
    }
};

int main(int argc, const char* argv[]) {
    cout << "Delphi REST API running!" << endl;

    Database* sqlite3DB = new Database();
    Experiment* experiment = new Experiment();

    served::multiplexer mux;


    /* what's the time? */
    time_t startTime = time(0);
    char* startTimeStr = ctime(&startTime);
    cout << "Start time (MST): " << startTimeStr;

    /* declare if CI is running, model creation will be shorter */
    if (getenv("CI")) {
        cout << "CI mode detected" << endl;
    }


    /* Allow users to stop computation of a running job */
    mux.handle("/stop_computation")
        .get([&sqlite3DB](served::response& res, const served::request& req) {
        cout << "ENDPOINT: /stop_computation" << endl; 
	res << "Delphi REST API 'stop_computation' called.";
    });

    /* Allow users to check if the REST API is running */
    mux.handle("/status")
        .get([&sqlite3DB](served::response& res, const served::request& req) {
        cout << "/status" << endl; 

        if (getenv("CI")) {
	    res << "The Delphi REST API is running in CI mode.";
        } else {
	    res << "The Delphi REST API is running.";
	}
    });


    /* Test the served response timing */
    mux.handle("/delphi/database/profiler")
        .post([&sqlite3DB](served::response& res, const served::request& req) {
        cout << "ENDPOINT: /delphi/database/profiler" << endl;

        Profiler* profiler = new Profiler();
        profiler->test();  // validate instantiation

        json ret_exp;

	delete profiler;

        return ret_exp;
    });






    /* openApi 3.0.0 
     * Post a new Delphi model. 
     */
    mux.handle("create-model")
        .post([&sqlite3DB](served::response& response, const served::request& req) {
            cout << "ENDPOINT: create-model" << endl;

            nlohmann::json json_data = nlohmann::json::parse(req.body());

	    // Find the id field, stop now if not foud.
	    string failEarly = "model ID not found";
	    string modelId = json_data.value("id",failEarly);
	    if(modelId == failEarly) {
	      cout << "Could not find 'id' field in input" << endl;
              json ret_exp;
              ret_exp["status"] = failEarly;
              response << ret_exp.dump();
              return ret_exp.dump(); 
	    }

            cout << "Model ID: " << modelId << endl;

	    // commented out for CI bug hunt, TODO: must be uncommented for production.
	    //sqlite3DB->init_training_status(modelId); 


            /* dump the input file to screen */
	    // cout << json_data.dump() << endl;

            /*
             If neither "CI" or "DELPHI_N_SAMPLES" is set, we default to a
             sampling resolution of 1000.

             TODO - we might want to set the default sampling resolution with
             some kind of heuristic, based on the number of nodes and edges. -
             Adarsh
            */
            size_t kde_kernels = 1000;
            int sampling_resolution = 1000, burn = 10000;
            if (getenv("CI")) {
	        cout << "CI mode detected" << endl;
                // When running in a continuous integration run, we set the
                // sampling resolution to be small to prevent timeouts.
                kde_kernels = 5;
                sampling_resolution = 5;
                burn = 5;
            }
            else if (getenv("DELPHI_N_SAMPLES")) {
                // We also enable setting the sampling resolution through the
                // environment variable "DELPHI_N_SAMPLES", for development and
                // testing purposes.
                kde_kernels = (size_t)stoul(getenv("DELPHI_N_SAMPLES"));
                sampling_resolution = 100;
                burn = 100;
            }

            AnalysisGraph G;
            G.set_res(kde_kernels);
            G.from_causemos_json_dict(json_data, 0, 0);

            sqlite3DB->insert_into_delphimodel(
                json_data["id"], G.serialize_to_json_string(false));

            auto response_json =
                nlohmann::json::parse(G.generate_create_model_response());

            try {
                thread executor_create_model(&Experiment::train_model,
                                             sqlite3DB,
                                             G,
                                             json_data["id"],
                                             sampling_resolution,
                                             burn);
                executor_create_model.detach();
		cout << "Training model " << modelId << endl;
            }
            catch (std::exception& e) {
                cout << "Error: unable to start training process" << endl;
		json no_train;
                no_train["status"] = "server error: training";
                response << no_train.dump();
                return no_train.dump();
            }

            // response << response_json.dump();
            // return response_json.dump();

            string strresult = response_json.dump();
            response << strresult;
            return strresult;
        });

    /* openApi 3.0.0
         """ Fetch experiment results"""
         Get asynchronous response while the experiment is being trained from
         causemosasyncexperimentresult table

         NOTE: I saw some weird behavior when we request results for an invalid
         experiment ID just after running an experiment. The trained model
       seemed to be not saved to the database. The model got re-trained from
       scratch on a subsequent experiment after the initial experiment and the
       invalid experiment result request. When I added a sleep between the
       initial create experiment and the invalid result request this re-training
       did not occur.
    */
    mux.handle("/models/{modelId}/experiments/{experimentId}")
        .get([&sqlite3DB](served::response& res, const served::request& req) {
            cout << "ENDPOINT: /models/{modelId}/experiments/{experimentId}" << endl;
            json result = sqlite3DB->select_causemosasyncexperimentresult_row(
                req.params["experimentId"]);
            if (result.empty()) {
                // experimentID not in database. Should be an incorrect
                // experimentID
                result["experimentType"] = "UNKNOWN";
                result["status"] = "invalid experiment id";
                result["results"] = "";
            }
            result["modelId"] = req.params["modelId"];
            result["experimentId"] = req.params["experimentId"];

            res << result.dump();
        });


    /* openApi 3.0.0
       Post a new causemos asynchronous experiment by creating
       a new thread to run the projection and returning
       served's REST response immediately.
    */
    mux.handle("/models/{modelId}/experiments")
        .post([&sqlite3DB](served::response& res, const served::request& req) {

            cout << "ENDPOINT: /models/{modelId}/experiments" << endl;

            auto request_body = nlohmann::json::parse(req.body());
	    string modelId = req.params["modelId"]; // should catch if not found

            json query_result = sqlite3DB->select_delphimodel_row(modelId);
            if (query_result.empty()) {
                json ret_exp;
                ret_exp["experimentId"] = "invalid model id";
                res << ret_exp.dump();
                return ret_exp;
            }

            string model = query_result["model"];

	    cout << "Model: " << model << endl;

            bool trained = nlohmann::json::parse(model)["trained"];

            if (trained == false) {
                json ret_exp;
                ret_exp["experimentId"] = "model not trained";
                res << ret_exp.dump();
                return ret_exp;
            }

            string experiment_type = request_body["experimentType"];
            boost::uuids::uuid uuid = boost::uuids::random_generator()();
            string experiment_id = to_string(uuid);

            sqlite3DB->insert_into_causemosasyncexperimentresult(
                experiment_id, "in progress", experiment_type, "");

            try {
                thread executor_experiment(&Experiment::runExperiment,
                                           sqlite3DB,
                                           req,
                                           modelId,
                                           experiment_id);
                executor_experiment.detach();
            }
            catch (std::exception& e) {
                cout << "Error: unable to start experiment process" << endl;

                json ret_exp;
                ret_exp["experimentId"] = "server error: experiment";
                res << ret_exp.dump();
                return ret_exp;
            }

            json ret_exp;
            ret_exp["modelId"] = modelId;
            ret_exp["experimentId"] = experiment_id;
            return ret_exp;
        });

    /* openApi 3.0.0
     * Query the training progress for a model.   
     */
    mux.handle("/models/{modelId}/training-progress")
        .get([&sqlite3DB](served::response& res, const served::request& req) {

	    cout << "ENDPOINT: /models/{modelId}/training-progress" << endl;

	    string modelId = req.params["modelId"]; // should catch missing

	    /*  commented out for CI bughunt.  TODO uncomment for production
            json query_result = sqlite3DB->select_training_status(modelId);
            if (query_result.empty()) {
	        string error = "training progress not found for id: " + modelId;
                json ret_exp;
                ret_exp["modelId"] = error;
                res << ret_exp.dump();
		return;
            }

            res << query_result.dump();
	    */

	    // used for CI bughunt.  TODO: Delete for production
	    json bughunt_exp;
            bughunt_exp["modelId"] = modelId;
            bughunt_exp["progress"] = "Not implented";
	    res << bughunt_exp.dump();
        });


    /* openApi 3.0.0
     * Post edge indicators for this model
     * TODO This needs to be implemented
     */
    mux.handle("/models/{modelId}/edit-indicators")
        .post([&sqlite3DB](served::response& res, const served::request& req) {
	    cout << "ENDPOINT: /models/{modelId}/edit-indicators" << endl;
	    string modelId = req.params["modelId"]; // should catch if not found
            json result = sqlite3DB->select_causemosasyncexperimentresult_row(modelId);

            if (result.empty()) {
                // model ID not in database. 
            }
            result["modelId"] = modelId;
            result["serverError"] = "edit-indicators not implemented";

            res << result.dump();
        });


    /* openApi 3.0.0
     * Post edge edits for this model
     * TODO This needs to be implemented
     */
    mux.handle("/models/{modelId}/edit-edges")
        .post([&sqlite3DB](served::response& res, const served::request& req) {
	    cout << "ENDPOINT: /models/{modelId}/edit-edges" << endl;
	    string modelId = req.params["modelId"]; // should catch if not found
            json result = sqlite3DB->select_causemosasyncexperimentresult_row(modelId);
            if (result.empty()) {
                // model ID not in database. 
            }
            result["modelId"] = modelId;
            result["serverError"] = "edit-edges not implemented";

            res << result.dump();
        });

    mux.handle("/models/{modelId}/training-stop")
        .get([&sqlite3DB](served::response& res, const served::request& req) {
	    cout << "ENDPOINT: /models/{modelId}/training-stop" << endl;
	    string modelId = req.params["modelId"]; // should catch if not found
            if (modelId.empty()) {
	        string error = "model ID not found: ";
                json ret_exp;
                ret_exp["modelId"] = error;
                res << ret_exp.dump();
		return;
                // model ID not in database. 
            }
            json ret_exp;
            ret_exp["modelId"] = modelId;
            ret_exp["serverError"] = "training-stop not implemented";

            res << ret_exp.dump();
        });


    /* openApi 3.0.0
     * Get the status of an existing model
     */
    mux.handle("/models/{modelId}")
        .get([&sqlite3DB](served::response& res, const served::request& req) {

            cout << "ENDPOINT: /models/{modelId}" << endl;

            string modelId = req.params["modelId"];

	    cout << "Checking status of model: " << modelId << endl;

            json query_result = sqlite3DB->select_delphimodel_row(modelId);

            if (query_result.empty()) {
                json ret_exp;
                ret_exp["status"] = "invalid model id";
                res << ret_exp.dump();
                return;
            }

            string model = query_result["model"];

            AnalysisGraph G;
            G = G.deserialize_from_json_string(model, false);
            auto response =
                nlohmann::json::parse(G.generate_create_model_response());

	    string status = response.value("status","Could not read status");

	    cout << status << endl;

            res << response.dump();
        });

    served::net::server server("127.0.0.1", "8123", mux);
    server.run(10);

    return (EXIT_SUCCESS);
}
