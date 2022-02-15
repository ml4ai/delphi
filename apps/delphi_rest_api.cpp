#include "AnalysisGraph.hpp"
#include "DatabaseHelper.hpp"
#include "ModelStatus.hpp"
#include "ExperimentStatus.hpp"
#include <assert.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/program_options.hpp>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/asio/ip/host_name.hpp>    // hostname
#include <ctime>
#include <iostream>
#include <nlohmann/json.hpp>
#include <range/v3/all.hpp>
#include <served/served.hpp>
#include <served/parameters.hpp>
#include <sqlite3.h>
#include <string>
#include <thread>

#ifdef TIME
    #include "Timer.hpp"
    #include "CSVWriter.hpp"
#endif

using namespace std;
using json = nlohmann::json;
namespace po = boost::program_options;

/*
    =====================
    Delphi - CauseMos API
    =====================
*/

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
            unordered_map<string, vector<string>> res_data;
            res_data["data"] = {};
            result["results"] = res_data;
            for (const auto& [conceptname, timestamp_sample_matrix] :
                 causemos_experiment_result) {
                json data_dict;
                data_dict["concept"] = conceptname;
                data_dict["values"] = vector<json>{};
                for (int i = 0; i < timestamp_sample_matrix.size(); i++) {
                    json time_series;
                    time_series["timestamp"] = timesteps_nparr[i];
                    time_series["values"] = timestamp_sample_matrix[i];
                    data_dict["values"].push_back(time_series);
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
	G.experiment_id = experiment_id;

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

};

class Model {

    public:

    static void train_model(
        Database* sqlite3DB,
        AnalysisGraph G,
	ModelStatus ms,
        string model_id,
        int res,
        int burn
    ) {
	G.id = model_id;
        G.run_train_model(res, burn);
	ms.set_status("trained, writing to database");
        sqlite3DB->insert_into_delphimodel(
            model_id,
            G.serialize_to_json_string(false)
        );
        ms.finalize("Ready");
	if(!ms.unlock()) {
	  cout << "Model " << model_id << " unlock failed" << endl;
	}
    }

    static size_t get_kde_kernels() {
        // When running in a continuous integration run, we set the
        // sampling resolution to be small to prevent timeouts.
        if (getenv("CI")) {
            return 200;
        }

        // We also enable setting the sampling resolution through the
        // environment variable "DELPHI_N_SAMPLES", for development and
        // testing purposes.
        if (getenv("DELPHI_N_SAMPLES")) {
            return (size_t)stoul(getenv("DELPHI_N_SAMPLES"));
        } 

	// If no specific environment is set, return a default value
	return 200;
    }


    /*
    TODO - we might want to set the default sampling resolution with
    some kind of heuristic, based on the number of nodes and edges. -
    Adarsh
    */
    static int get_sampling_resolution() {
        return 100; 
    }

    static int get_burn() {
        // When running in a continuous integration run, we set the
        // burn to a low value to prevent timeouts.
        if (getenv("CI")) {
            return 1000;
        }
	
	// if the environment variable "DELPHI_N_SAMPLES" has been set,
	// we adjust the burn rate as well.
        if (getenv("DELPHI_N_SAMPLES")) {
            return 100;
	}

	// If no specific environment is set, return a default value
	return 10000;
    }

    // freeze the edge and return a status string (Empty means OK)
    static string freeze_edge(
        AnalysisGraph& G,
        string source_name,
        string target_name,
        double scaled_weight,
        int polarity
    ){
        switch (G.freeze_edge_weight(
            source_name,
	    target_name,
	    scaled_weight,
	    polarity
        )) {
            case 1: return "scaled_weight " 
                + to_string(scaled_weight) 
                + " outside accepted range";
            case 2: return "Source concept '" 
                + source_name
                + "' does not exist";
            case 4: return "Target concept '"
                + target_name
                + "' does not exist";
            case 8: return "There is no edge from '" 
                + source_name 
                + "' to '" 
                + target_name 
                + "'";
            default: return "";
	}
    }
};


// System runtime parameters returned by the 'status' endpoint
std::string getSystemStatus() {

    // report the host machine name
    const auto host_name = boost::asio::ip::host_name(); 

    // report the run mode.  CI is used for debugging.
    const auto run_mode = getenv("CI") ? "CI" : "normal";

    // report the start time
    char timebuf[200];
    time_t t;
    struct tm *now;
    const char* fmt = "%F %T";
    t = time(NULL);
    now = gmtime(&t);
    if (now == NULL) {
      perror("gmtime error");
      exit(EXIT_FAILURE);
    }
    if (strftime(timebuf, sizeof(timebuf), fmt, now) == 0) {
      fprintf(stderr, "strftime returned 0");
      exit(EXIT_FAILURE);
    }

    std::string system_status = "The Delphi REST API was started on "
      + host_name
      + " in "
      + run_mode
      + " mode at UTC "
      + timebuf;

    return system_status;
}

// the system status only has to be generated once.
string systemStatus = getSystemStatus();


int main(int argc, const char* argv[]) {
    // Declare the supported options.
    po::options_description desc("Allowed options");
    string host;
    int port;
    desc.add_options()("help,h", "produce help message")(
        "host",
        po::value<string>(&host)->default_value("localhost"),
        "Set host")(
        "port", po::value<int>(&port)->default_value(8123), "Set port");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 1;
    }

    // report status on startup
    cout << systemStatus << endl;

    Database* sqlite3DB = new Database();
    Experiment* experiment = new Experiment();
    served::multiplexer mux;

    // prepare the model and experiment databases for use
    ModelStatus ms("startup", sqlite3DB);
    ExperimentStatus es("startup", "startup", sqlite3DB);
    ms.clean_db();
    es.clean_db();

    /* Allow users to check if the REST API is running */
    mux.handle("/status").get(
        [&sqlite3DB](served::response& res, const served::request& req) {
	    // report status on request
	    res << systemStatus;
        });

    /* openApi 3.0.0
     * Post a new Delphi model.
     */
    mux.handle("create-model")
        .post([&sqlite3DB](served::response& response,
                           const served::request& req) {
            nlohmann::json req_json = nlohmann::json::parse(req.body());

	    // input must have a model ID field, which is "id" per the API
	    if(!req_json.contains("id")) {
                json ret;
		ret["id"] = "Not found";
                ret["status"] = "Model input must contain an 'id' field";
                response << ret.dump();
                return ret.dump();
            }

	    string modelId = req_json["id"];
	    ModelStatus ms(modelId, sqlite3DB);

            // do not overwrite model if it is training
            if (!ms.lock()) {
		json model_data = ms.get_data();
                json ret;
                ret[ms.MODEL_ID] = modelId;
                ret[ms.PROGRESS] = model_data[ms.PROGRESS];
                ret[ms.STATUS] = "Model is busy(" 
                    + (string)model_data[ms.STATUS] 
		    + "), please wait until it before overwriting.";
                string dumpStr = ret.dump();
                response << dumpStr; 
                return dumpStr; 
            }
	    
	    // create the model on the database
            ms.set_status("Creating Model");

	    size_t kde_kernels = Model::get_kde_kernels();
	    int burn = Model::get_burn();
	    int sampling_resolution = Model::get_sampling_resolution();

            AnalysisGraph G;
            G.set_n_kde_kernels(kde_kernels);
            G.from_causemos_json_dict(req_json, 0, 0);

            ms.set_status("Writing untrained Model to database");
            sqlite3DB->insert_into_delphimodel(
                modelId, 
		G.serialize_to_json_string(false)
            );

            auto response_json =
                nlohmann::json::parse(G.generate_create_model_response());

            try {
                thread executor_create_model(&Model::train_model,
                                             sqlite3DB,
                                             G,
					     ms,
					     modelId,
                                             sampling_resolution,
                                             burn);
                executor_create_model.detach();
            }
            catch (std::exception& e) {
                cout << "Error: unable to start training process" << endl;
                json error;
                error["status"] = "server error: training";
                response << error.dump();
                return error.dump();
            }

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

            string modelId = req.params["modelId"];
            string experimentId = req.params["experimentId"];

	    ExperimentStatus es(modelId, experimentId, sqlite3DB);

            json query_result =
                sqlite3DB->select_causemosasyncexperimentresult_row(
                    experimentId
		);

	    json ret;
            ret[es.MODEL_ID] = modelId;
            ret[es.EXPERIMENT_ID] = experimentId;
            ret[es.EXPERIMENT_TYPE] = query_result[es.EXPERIMENT_TYPE];
            ret[es.STATUS] = query_result[es.STATUS];
            ret[es.PROGRESS] = query_result[es.PROGRESS];
            ret[es.RESULTS] = query_result["data"];

	    res << ret.dump();
            return ret;
        });

    /* openApi 3.0.0
       Post a new causemos asynchronous experiment by creating
       a new thread to run the projection and returning
       served's REST response immediately.
    */
    mux.handle("/models/{modelId}/experiments")
        .post([&sqlite3DB](served::response& res, const served::request& req) {
            auto request_body = nlohmann::json::parse(req.body());
            string modelId = req.params["modelId"]; 

            json ret;
            ret["modelId"] = modelId;

	    ModelStatus ms(modelId, sqlite3DB);
	    json model_data = ms.get_data();

	    // Model not found
	    if(model_data.empty()) {
                ret[ms.STATUS] = "Invalid model ID";
                res << ret.dump();
                return ret;
	    }

            // Model busy
	    bool model_busy = model_data[ms.BUSY];
	    if(model_busy) {
                ret[ms.PROGRESS] = model_data[ms.PROGRESS];
                ret[ms.STATUS] = "Model is busy(" 
                    + (string)model_data[ms.STATUS] 
                    + "), please wait until it finishes before overwriting.";
                res << ret.dump();
                return ret;
            }

            string experiment_type = request_body["experimentType"];
            boost::uuids::uuid uuid = boost::uuids::random_generator()();
            string experiment_id = to_string(uuid);

	    ExperimentStatus es(experiment_id, modelId, sqlite3DB);

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
		string report = "Error: unable to start experiment process";
                cout << report << endl;
                ret[es.STATUS] = report;
                res << ret.dump();
                return ret;
            }

            ret[es.EXPERIMENT_ID] = experiment_id;

            res << ret.dump();
            return ret;
        });

    /* openApi 3.0.0
     * Query the training progress for a model.
     */
    mux.handle("/models/{modelId}/training-progress").get([&sqlite3DB](
        served::response& res,
        const served::request& req
    ){
        string modelId = req.params["modelId"];

        ModelStatus ms(modelId, sqlite3DB);
	json model_data = ms.get_data();
        json ret;
        ret[ms.MODEL_ID] = modelId;

	if(model_data.empty()) {
            ret[ms.STATUS] = "Model ID not found";
            res << ret.dump();
	    return;
        }

        ret[ms.PROGRESS] = model_data[ms.PROGRESS];
        ret[ms.STATUS] = model_data[ms.STATUS];
        ret[ms.BUSY] = model_data[ms.BUSY];
        res << ret.dump();
    });

    /* openApi 3.0.0
     * Post edge indicators for this model
     * TODO This needs to be implemented
     */
    mux.handle("/models/{modelId}/edit-indicators")
        .post([&sqlite3DB](served::response& res, const served::request& req) {
            string message =
                "The edit-indicators endpoint is not implemented for Delphi, "
                "since Delphi (as it is currently implemented) needs to "
                "retrain the whole model on every indicator change. This might "
                "change in the future, but for now, the way to update an "
                "existing model is to use the create-model API endpoint.";

            string modelId = req.params["modelId"]; 
	    ModelStatus ms(modelId, sqlite3DB);

            json ret;
            ret[ms.MODEL_ID] = modelId;
            ret[ms.STATUS] = message;
            res << ret.dump();
        });

    /* openApi 3.0.0
     * Post edge edits for this model
     * TODO This needs to be implemented
     */
    mux.handle("/models/{modelId}/edit-edges")

        .post([&sqlite3DB](served::response& res, const served::request& req) {

        nlohmann::json req_json = nlohmann::json::parse(req.body());

            string modelId = req.params["modelId"];
	    ModelStatus ms(modelId, sqlite3DB);
            json ret;
            ret[ms.MODEL_ID] = modelId;

	    auto relations = req_json["relations"];

	    // test input for edges
            if(relations.empty()) {
                // return error
                ret[ms.STATUS] = "Edges not found in input";
                res << ret.dump();
                return ret;
            }

	    json model_data = ms.get_data();

            // Model not found
            if(model_data.empty()) {
                ret[ms.STATUS] = "Model does not exist.";
                res << ret.dump();
                return ret;
            }   

            // See if this model is available for training
	    if(!ms.lock()) {
                ret[ms.PROGRESS] = model_data[ms.PROGRESS];
                ret[ms.STATUS] = "Model is busy(" 
                    + (string)model_data[ms.STATUS] 
		    + "), please wait until it finished before editing.";
                res << ret.dump();
                return ret;
            }  

	    // beyond this point we have exclusive control of this model
            ms.set_status("Deserializing model");
            
	    // Get the model row from the database
            json query_result = sqlite3DB->select_delphimodel_row(modelId);

	    // deserialize the model
	    string modelString = query_result["model"];
            AnalysisGraph G;
            G = G.deserialize_from_json_string(modelString, false);

            ms.set_status("Freezing edges");

	    // freeze edges
	    for(auto relation : relations) {
		string source = relation["source"];
		string target = relation["target"];
		vector<double> weights = relation["weights"];
		double weight = weights.front();
		int polarity = relation["polarity"];

		// test the return value
		string errorReport = 
                    Model::freeze_edge(G, source, target, weight, polarity);
		if(!errorReport.empty()) {
                    ret[ms.STATUS] = errorReport;
                    res << ret.dump();
                    return ret;
		}
	    }


	    // train model in another thread
	    size_t kde_kernels = Model::get_kde_kernels();
	    int burn = Model::get_burn();
	    int sampling_resolution = Model::get_sampling_resolution();

            G.set_n_kde_kernels(kde_kernels);

            sqlite3DB->insert_into_delphimodel(
                modelId,
                G.serialize_to_json_string(false)
            );

            try {
                thread executor_create_model(&Model::train_model,
                                             sqlite3DB,
                                             G,
					     ms,
                                             modelId,
                                             sampling_resolution,
                                             burn);
                executor_create_model.detach();
                cout << "Training model " << modelId << endl;
            }
            catch (std::exception& e) {
                cout << "Error: unable to start training process" << endl;
                json error;
                error["status"] = "server error: edged-edges training";
                res << error.dump();
                return error;
            }
	   
	    // report success 
	    ret[ms.STATUS] = "Edges frozen: "
              + to_string(relations.size())
              + ". Model is in training.";
            res << ret.dump();
            return ret;
        });


    /* openApi 3.0.0
     * Get the status of an existing model
     */
    mux.handle("/models/{modelId}")
        .get([&sqlite3DB](served::response& res, const served::request& req) {

            string modelId = req.params["modelId"];

            #ifdef TIME
                CSVWriter writer = CSVWriter(string("timing") + "_" +
                                             modelId + "_" +
                                             "model_status" + "_" +
                                             delphi::utils::get_timestamp() +
                                             ".csv");
                vector<string> headings = {"Query DB WC Time (ns)",
                                           "Query DB CPU Time (ns)",
                                            "Deserialize WC Time (ns)",
                                            "Deserialize CPU Time (ns)",
                                            "Response WC Time (ns)",
                                            "Response CPU Time (ns)"};
                writer.write_row(headings.begin(), headings.end());
                std::pair<std::vector<std::string>, std::vector<long>> durations;
                durations.second.clear();
            #endif
            json query_result;
            {
                #ifdef TIME
                    Timer t = Timer("Query", durations);
                #endif
                query_result = sqlite3DB->select_delphimodel_row(modelId);
            }

            if (query_result.empty()) {
                json ret_exp;
                ret_exp["status"] = "Invalid model ID";
                res << ret_exp.dump();
                return;
            }

            string model = query_result["model"];

            AnalysisGraph G;
            {
                #ifdef TIME
                    Timer t = Timer("Deserialize", durations);
                #endif
                G = G.deserialize_from_json_string(model, false);
            }

            {
                #ifdef TIME
                    Timer t = Timer("Response", durations);
                #endif
                auto response =
                    nlohmann::json::parse(G.generate_create_model_response());

                res << response.dump();
            }
            #ifdef TIME
                writer.write_row(durations.second.begin(), durations.second.end());
                cout << "\nElapsed time to deserialize (seconds): " << durations.second[2] / 1000000000.0 << endl;
            #endif
        });

    served::net::server server(host, to_string(port), mux);
    server.run(10);

    return (EXIT_SUCCESS);
}
