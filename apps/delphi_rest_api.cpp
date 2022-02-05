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
        string modelId,
        int sampling_resolution,
        int burn
    ) {
        G.run_train_model(
            sampling_resolution,
            burn,
            InitialBeta::ZERO,
            InitialDerivative::DERI_ZERO);
        sqlite3DB->insert_into_delphimodel(
            modelId,
            G.serialize_to_json_string(false));
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

    Database* sqlite3DB = new Database();
    Experiment* experiment = new Experiment();
    ModelStatus ms;
    ExperimentStatus es;
    served::multiplexer mux;

    ms.init_db();
    es.init_db();

    // report status on startup
    cout << systemStatus << endl;

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
            nlohmann::json json_data = nlohmann::json::parse(req.body());
	    ModelStatus ms(sqlite3DB);

	    // input must have a model ID field
	    if(!json_data.contains(ms.MODEL_ID)) {
	        string report = "Input must contain an '" 
		  + ms.MODEL_ID 
		  + "' field.";
                json ret_exp;
                ret_exp[ms.MODEL_ID] = "Not found";
                ret_exp[ms.STATUS] = report;
                response << ret_exp.dump();
                return ret_exp.dump();
            }

	    string modelId = json_data[ms.MODEL_ID];

            // If this model does not exist, set initial status state
	    json status = ms.get_status(modelId);
            if(status.empty()) {
                ms.set_initial_status(modelId);
            }
            // If this model does exist, do not overwrite if training.
            else {
                bool trained = status[ms.TRAINED];
                if(!trained) {
                  json ret;
                  ret[ms.MODEL_ID] = modelId;
                  ret[ms.PROGRESS] = status[ms.PROGRESS];
                  ret[ms.STATUS] = "A model with the same ID is still training";
		  string dumpStr = ret.dump();
                  response << dumpStr; 
                  return dumpStr; 
                }     
            }

	    size_t kde_kernels = Model::get_kde_kernels();
	    int burn = Model::get_burn();
	    int sampling_resolution = Model::get_sampling_resolution();

            AnalysisGraph G;
            G.set_n_kde_kernels(kde_kernels);
            G.from_causemos_json_dict(json_data, 0, 0);

            sqlite3DB->insert_into_delphimodel(
                json_data["id"], G.serialize_to_json_string(false));

            auto response_json =
                nlohmann::json::parse(G.generate_create_model_response());

            try {
                thread executor_create_model(&Model::train_model,
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
                json error;
                error["status"] = "server error: training";
                response << error.dump();
                return error.dump();
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

            string modelId = req.params["modelId"];
            string experimentId = req.params["experimentId"];

	    ExperimentStatus es(sqlite3DB);

            json query_result =
                sqlite3DB->select_causemosasyncexperimentresult_row(
                    experimentId
		);

	    json ret;
            ret[es.MODEL_ID] = modelId;
            ret[es.EXPERIMENT_ID] = experimentId;

            if (query_result.empty()) { // should not be possible?
                ret[es.EXPERIMENT_TYPE] = "UNKNOWN";
                ret[es.STATUS] = "invalid experiment id";
                ret[es.RESULTS] = "Not found";
            } else {
                string resultstr = query_result[es.RESULTS];
                json results = json::parse(resultstr);
                ret[es.EXPERIMENT_TYPE] = query_result[es.EXPERIMENT_TYPE];
                ret[es.STATUS] = query_result[es.STATUS];
                ret[es.PROGRESS] = query_result[es.PROGRESS];
                ret[es.RESULTS] = results["data"];
            } 

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
            string modelId = req.params["modelId"]; // should catch if not found

	    ModelStatus ms(sqlite3DB);
	    ExperimentStatus es(sqlite3DB);

            json ret;
            ret[es.MODEL_ID] = modelId;

	    json status = ms.get_status(modelId);

	    // Model not found
	    if(status.empty()) {
              ret[es.STATUS] = "Invalid model ID";
              res << ret.dump();
              return ret;
	    }

            // Model not trained
	    bool trained = status[ms.TRAINED];
	    if(!trained) {
	      ret[es.STATUS] = "Model training must finish before experimenting";
              res << ret.dump();
              return ret;
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
        ModelStatus ms(sqlite3DB);
        string modelId = req.params["modelId"];
	json status = ms.get_status(modelId);
        json ret;
        ret[ms.MODEL_ID] = modelId;
	if(status.empty()) {
            ret[ms.STATUS] = "Invalid model ID";  // Model ID not found
	} else {
	    ret[ms.TRAINED] = status[ms.TRAINED];
	    ret[ms.PROGRESS] = status[ms.PROGRESS];
	}
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

            string modelId = req.params["modelId"]; // should catch if not found
            json result =
                sqlite3DB->select_causemosasyncexperimentresult_row(modelId);

            if (result.empty()) {
                // model ID not in database.
            }
            result["modelId"] = modelId;
            result["status"] = message;

            res << result.dump();
        });

    /* openApi 3.0.0
     * Post edge edits for this model
     * TODO This needs to be implemented
     */
    mux.handle("/models/{modelId}/edit-edges")

        .post([&sqlite3DB](served::response& res, const served::request& req) {

        nlohmann::json req_json = nlohmann::json::parse(req.body());


	    ModelStatus ms(sqlite3DB);
            string modelId = req.params["modelId"];
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

	    // Get the model row from the database
            json query_result = sqlite3DB->select_delphimodel_row(modelId);

	    // if nothing is found, the model ID is invalid
	    if(query_result.empty()) {
              ret[ms.STATUS] = "Invalid model ID";
              res << ret.dump();
              return ret;
	    }

	    // deserialize the model
	    string modelString = query_result["model"];
            AnalysisGraph G;
            G = G.deserialize_from_json_string(modelString, false);

            // Return an error if the model is not trained.
	    if(!G.get_trained()) {
	      ret[ms.TRAINED] = false;
	      ret[ms.STATUS] = "Training must finish before editing edges";
              res << ret.dump();
              return ret;
            }

	    // freeze edges
	    for(auto relation : relations) {
	        cout << "Freezing edge:" << endl;

		string source = relation["source"];
		cout << "  Source = " << source << endl;

		string target = relation["target"];
		cout << "  Target = " << target << endl;

		int polarity = relation["polarity"];
		cout << "  Polarity = " << polarity << endl;

		vector<double> weights = relation["weights"];
		double weight = weights.front();

		cout << "  Weight = " << weight << endl;
		G.freeze_edge_weight(source, target, weight, polarity);
	    }

	    // train model in another thread
	    size_t kde_kernels = Model::get_kde_kernels();
	    int burn = Model::get_burn();
	    int sampling_resolution = Model::get_sampling_resolution();

            G.set_n_kde_kernels(kde_kernels);

            try {
                thread executor_create_model(&Model::train_model,
                                             sqlite3DB,
                                             G,
                                             modelId,
                                             sampling_resolution,
                                             burn);
                executor_create_model.detach();
                cout << "Training model " << modelId << endl;
            }
            catch (std::exception& e) {
                cout << "Error: unable to start training process" << endl;
                json error;
                error["status"] = "server error: training";
                res << error.dump();
                return error;
            }
	   
	    // report success 
	    int nEdges = relations.size();
	    char buf[200];
	    if(nEdges == 1) {
	      sprintf(buf, "1 edge");
	    } else {
	      sprintf(buf, "%d edges", nEdges);
	    }
	    ret[ms.STATUS] = string(buf) + " frozen, model is in training.";
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
