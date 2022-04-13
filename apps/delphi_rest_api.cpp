#include "AnalysisGraph.hpp"
#include "DatabaseHelper.hpp"
#include "ModelStatus.hpp"
#include "Logger.hpp"
#include "Config.hpp"
#include "ExperimentStatus.hpp"
#include <assert.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/program_options.hpp>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/asio/ip/host_name.hpp>    // hostname
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

        ExperimentStatus es(experiment_id, modelId, sqlite3DB);

        if (!trained) {
            if(!sqlite3DB->insert_into_delphimodel(
                modelId, G.serialize_to_json_string(false))) {
		es.enter_finished_state("Could not insert model into database");
		return;
	    }
        }

        json result =
            sqlite3DB->select_causemosasyncexperimentresult_row(experiment_id);

	if(result.empty()) {
            es.enter_finished_state("Could find experiment in database");
	    return;  // experiment not found
	}
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

	if(!sqlite3DB->insert_into_causemosasyncexperimentresult(
            result["id"],
            result["status"],
            result["experimentType"],
            result["results"].dump()
        )) {
            es.enter_finished_state(
                "Unable to insert experiment into database"
            );
	}
	else {
            es.enter_finished_state((string)result["status"]);
	}
    }

    static void runExperiment(Database* sqlite3DB,
                              const served::request& request,
                              string modelId,
                              string experiment_id) {
        auto request_body = nlohmann::json::parse(request.body());
        string experiment_type = request_body["experimentType"];

	ModelStatus ms(modelId, sqlite3DB);
	ExperimentStatus es(experiment_id, modelId, sqlite3DB);

        json query_result = sqlite3DB->select_delphimodel_row(modelId);

        if (ms.read_data().empty()) {
            // Model ID not in database.
            query_result = sqlite3DB->select_causemosasyncexperimentresult_row(
                experiment_id);
            if(!sqlite3DB->insert_into_causemosasyncexperimentresult(
                query_result["id"],
                "failed",
                query_result["experimentType"],
                query_result["results"])
             ) {
                es.enter_finished_state(
                    "Unable to insert experiment into database"
                );
		return;
	    }

        }

	es.enter_reading_state();

        string model = query_result["model"];
        bool trained = nlohmann::json::parse(model)["trained"];

        AnalysisGraph G;
        G = G.deserialize_from_json_string(model, false);
	G.experiment_id = experiment_id;

	if(experiment_type == "PROJECTION") {
            runProjectionExperiment(
                sqlite3DB, request, modelId, experiment_id, G, trained
            );
	}

        /* experiment types not implemented:
        GOAL_OPTIMIZATION
        SENSITIVITY_ANALYSIS
        MODEL_VALIDATION
        BACKCASTING
        */
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
        if(!sqlite3DB->insert_into_delphimodel(
            model_id,
            G.serialize_to_json_string(false)
        )) {
            ms.enter_finished_state(
                "Error, unable to insert trained model into database"
            );
	    return;
	} 
	ms.enter_finished_state("Trained");
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
	Config config;
	return config.get_int("training_burn", 1000);
    }
};

// System runtime parameters returned by the 'status' endpoint
std::string getSystemStatus() {

  // report the host machine name
  const auto host_name = boost::asio::ip::host_name(); 

  // report the run mode.  CI is used for debugging.
  const auto run_mode = getenv("CI") ? "CI" : "normal";

  // report the start time
  timeval curTime;
  gettimeofday(&curTime, NULL);
  int milli = curTime.tv_usec / 1000;

  char buffer [80];
  strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&curTime.tv_sec));

  char current_time[84] = "";
  sprintf(current_time, "%s:%03d", buffer, milli);

  std::string system_status = "The Delphi REST API was started on "
    + host_name
    + " in "
    + run_mode
    + " mode at UTC "
    + current_time;

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

    // start new logfile on startup
    Logger logger;

    // report status
    logger.overwrite_logfile(systemStatus);
    cout << systemStatus << endl;

    // check database tables.  Showstopper if none are found.
    Database* sqlite3DB = new Database();
    string query = "select name from sqlite_master where type='table';";
    vector<string> results = sqlite3DB->read_column_text(query);
    char *db_path = getenv ("DELPHI_DB");
    if(results.empty()) {
        string db_report = "Could not find Delphi database at "
            + string(db_path);
        logger.error(db_report);
        cerr << db_report << endl;
        return 1;
    } 
    string db_report = "Using Delphi database at "
        + string(db_path);
    logger.info(db_report);
    cout << db_report << endl;

    // report location of config file
    Config config;
    string fallback = "not_found";
    string label = "delphi_rest_api";
    string config_version = config.get_string("config_version", fallback);
    if(config_version == fallback) {
      string path = config.get_config_file_path();
      string report = "Could not locate configuration file '" + path + "'";
      cerr << report << endl;
      logger.error(report);
    } else {
      string report = "Using configuration version " + config_version;
      cout << report << endl;
      logger.info(report);
    }

    // initialize Delphi database and REST API endpoints
    Experiment* experiment = new Experiment();
    served::multiplexer mux;

    // prepare the model and experiment databases for use
    ModelStatus ms("startup", sqlite3DB);
    ExperimentStatus es("startup", "startup", sqlite3DB);
    ms.initialize();
    es.initialize();

    /* Allow users to check if the REST API is running */
    mux.handle("/status")
        .get([&sqlite3DB](served::response& res, const served::request& req) {
	    Logger logger;
	    logger.info("");
            logger.info("### DELPHI ENDPOINT: status");
	    logger.info(" " + systemStatus);
	    res << systemStatus;
        });

    /* openApi 3.0.0
     * Post a new Delphi model.
     */
    mux.handle("/create-model")
        .post([&sqlite3DB](served::response& response,
                           const served::request& req) {
			
	    Logger logger;
	    logger.info("");
            logger.info("### DELPHI ENDPOINT: create-model");

	    json ret;

            // no file uploaded
            if(req.body().empty()) {
                string error = "Error: No model data was received";
                ret["status"] = error;
                string dump = ret.dump();
                logger.error(" " + dump);
                response << dump;
                return ret.dump();
            }

            nlohmann::json req_json = nlohmann::json::parse(req.body());

	    // input must have a model ID field, which is "id" per the API
	    if(!req_json.contains("id")) {
                json ret;
		ret["id"] = "Not found";
                ret["status"] = "Model input must contain an 'id' field";
		string dump = ret.dump();
                logger.error(" " + dump);
                response << dump;
                return dump;
            }

	    string modelId = req_json["id"];
	    ModelStatus ms(modelId, sqlite3DB);

	    logger.info("model id = " + modelId);
           
	    ret[ms.MODEL_ID] = modelId;

	    // check for existence of row
	    json row = sqlite3DB->select_delphimodel_row(modelId);
	    if(row.empty()) {
		logger.info("Row not found for " + modelId + ", creating it...");
		// create it if needed
                string query = 
                    "INSERT into delphimodel ('id', 'model', 'progress') VALUES ('"
		    + modelId 
		    + "', 'empty', 'empty');";
		logger.info("Query = " + query);
		if(!sqlite3DB->insert(query)) {
                    ret[ms.STATUS] = "Model row could not be created ";
                    string dump = ret.dump();
                    response << dump; 
	            logger.error(" " + dump);
                    return dump; 
                }
                ms.enter_initial_state();
            } else {
               logger.info("Row exists for " +  modelId);
	    }

            json model_status_json = ms.read_data();
	    bool model_busy = model_status_json[ms.BUSY];

            // do not overwrite model if it is busy
            if (model_busy) {
                ret[ms.PROGRESS] = model_status_json[ms.PROGRESS];
                ret[ms.STATUS] = "Model is busy (" 
                    + (string)model_status_json[ms.STATUS] 
		    + "), please wait until it before overwriting.";
                string dump = ret.dump();
                response << dump; 
	        logger.warning(" " + dump);
                return dump; 
            }
	    
	    // create the model on the database
            ms.enter_reading_state();

	    size_t kde_kernels = Model::get_kde_kernels();
	    int burn = Model::get_burn();
	    int sampling_resolution = Model::get_sampling_resolution();

            AnalysisGraph G;
            G.set_n_kde_kernels(kde_kernels);
            G.from_causemos_json_dict(req_json, 0, 0);

            if(!sqlite3DB->insert_into_delphimodel(
                modelId, 
		G.serialize_to_json_string(false)
            )) {
                json error;
                error["status"] = "server error: unable to insert into delphimodel";
		string dump = error.dump();
	        logger.error(" " + dump);
                response << dump;
                return dump;
	    }

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
		string dump = error.dump();
	        logger.error(" " + dump);
                response << dump;
                return dump;
            }

            string dump = response_json.dump();
	    logger.info(" " + dump);
            response << dump;
            return dump;
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

	    Logger logger;
	    logger.info("");
	    logger.info("### DELPHI ENDPOINT: models/" 
	        + modelId 
		+ "/experiments/"
	       	+ experimentId);

            ExperimentStatus es(experimentId, modelId, sqlite3DB);

            json ret;
            ret[es.EXPERIMENT_ID] = experimentId;

	    json es_data = es.read_data();
	    if(!es_data.empty()) {
	      bool busy = es_data[es.BUSY];
	      if(busy) {
	        ret[es.PROGRESS] = es_data[es.PROGRESS];
	        ret[es.STATUS] = 
		  "Experiment is busy, please wait for it to finish.";
		string dump = ret.dump();
	        logger.warning(" " + dump);
                res << dump;
                return ret;
              }
            }

            json query_result =
                sqlite3DB->select_causemosasyncexperimentresult_row(
                    experimentId
                );

	    if (query_result.empty()) { // bad modelId or experimentId
                ret[es.STATUS] = "invalid experiment id";
                ret[es.RESULTS] = "Not found";
                string dump = ret.dump();
                logger.warning(" " + dump);
                res << dump;
                return ret;
            } 

            string resultstr = query_result[es.RESULTS];
            json results = json::parse(resultstr);
            double progress = es_data.value(es.PROGRESS, 0.0);
            ret[es.EXPERIMENT_TYPE] = query_result[es.EXPERIMENT_TYPE];
            ret[es.STATUS] = query_result[es.STATUS];
            ret[es.PROGRESS] = progress;
            ret[es.RESULTS] = results["data"];
	    
            string dump = ret.dump();
            logger.info(" " + dump);
            res << dump;
            return ret;
        });

    /* openApi 3.0.0
       Post a new causemos asynchronous experiment by creating
       a new thread to run the projection and returning
       served's REST response immediately.
    */
    mux.handle("/models/{modelId}/experiments")
        .post([&sqlite3DB](served::response& res, const served::request& req) {

            string modelId = req.params["modelId"]; 

            json ret;
            ret["modelId"] = modelId;

	    Logger logger;
            logger.info("");
            logger.info("### DELPHI ENDPOINT: models/" 
	        + modelId 
		+ "/experiments");
			
	    ModelStatus ms(modelId, sqlite3DB);

            // no file uploaded
            if(req.body().empty()) {
                string error = "Error: No experiment data was received";
                ret[ms.STATUS] = error;
                string dump = ret.dump();
                logger.error(" " + dump);
                res << dump;
                return ret;
            }

            auto request_body = nlohmann::json::parse(req.body());

	    // Model not found
	    json model_status_json = ms.read_data();
	    if(model_status_json.empty()) {
                ret[ms.STATUS] = "Invalid model ID";
		string dump = ret.dump();
		logger.error(" " + dump);
                res << dump;
                return ret;
	    }

            // Model busy
	    bool model_busy = model_status_json[ms.BUSY];
	    if(model_busy) {
                ret[ms.PROGRESS] = model_status_json[ms.PROGRESS];
                ret[ms.STATUS] = "Model is busy(" 
                    + (string)model_status_json[ms.STATUS] 
                    + "), please wait until it finishes before overwriting.";
		string dump = ret.dump();
		logger.warning(" " + dump);
                res << dump;
                return ret;
            }

            string experiment_type = request_body["experimentType"];
            boost::uuids::uuid uuid = boost::uuids::random_generator()();
            string experiment_id = to_string(uuid);

	    ExperimentStatus es(experiment_id, modelId, sqlite3DB);
	    es.enter_initial_state();
	    es.enter_reading_state();

            if(!sqlite3DB->insert_into_causemosasyncexperimentresult(
                experiment_id, "in progress", experiment_type, "")
            ) {
                string report = "Unable to insert experiment into database";
                es.enter_finished_state(report);
                ret[es.STATUS] = report;
		string dump = ret.dump();
		logger.error(" " + dump);
                res << dump;
                return ret;
	    }

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
                es.enter_finished_state(report);
                cout << report << endl;
                ret[es.STATUS] = report;
		string dump = ret.dump();
		logger.error(" " + dump);
                res << dump;
                return ret;
            }

            ret[es.EXPERIMENT_ID] = experiment_id;
            string dump = ret.dump();
            logger.info(" " + dump);
            res << dump;
            return ret;
        });

    /* openApi 3.0.0
     * Query the training progress for a model.
     */
    mux.handle("/models/{modelId}/training-progress")
        .get([&sqlite3DB](served::response& res, const served::request& req){
            string modelId = req.params["modelId"];

            Logger logger;
	    logger.info("");
            logger.info("### DELPHI ENDPOINT: models/" 
                + modelId 
                + "/training-progress");

            ModelStatus ms(modelId, sqlite3DB);
            json model_status_json = ms.read_data();
            json ret;
            ret[ms.MODEL_ID] = modelId;

            if(model_status_json.empty()) {
                ret[ms.STATUS] = "Model ID not found";
                string dump = ret.dump();
                logger.error(" " + dump);
                res << dump;
                return;
            }

            ret[ms.PROGRESS] = model_status_json[ms.PROGRESS];
            ret[ms.STATUS] = model_status_json[ms.STATUS];
            string dump = ret.dump();
            logger.info(" " + dump);
            res << dump;
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

            Logger logger;
            logger.info("");
            logger.info("### DELPHI ENDPOINT: models/" 
                + modelId 
                + "/edit-indicators");

            json ret;
            ret[ms.MODEL_ID] = modelId;

            if(req.body().empty()) {
                string error = "Error: No indicators data was received";
                ret[ms.STATUS] = error;
                logger.error(" " + ret.dump());
            }
            else {
                ret[ms.STATUS] = message;
                logger.warning(" " + ret.dump());
            }
            string dump = ret.dump();
            logger.info(" " + dump);
            res << dump;
        });

    /* openApi 3.0.0
     * Post edge edits for this model
     * TODO This needs to be implemented
     */
    mux.handle("/models/{modelId}/edit-edges")
        .post([&sqlite3DB](served::response& res, const served::request& req) {

            string modelId = req.params["modelId"];

            ModelStatus ms(modelId, sqlite3DB);

            Logger logger;
            logger.info("");
            logger.info("### DELPHI ENDPOINT: models/" 
                + modelId 
                + "/edit-edges");

            json ret;
            ret[ms.MODEL_ID] = modelId;

            // no file uploaded
            if(req.body().empty()) {
                string error = "Error: No edges data was received";
                ret[ms.STATUS] = error;
                string dump = ret.dump();
                logger.error(" " + dump);
                res << dump;
                return ret;
            }

            nlohmann::json req_json = nlohmann::json::parse(req.body());
            logger.info(" " + req_json.dump());

            json model_status_json = ms.read_data();

            // Model not found
            if(model_status_json.empty()) {
                ret[ms.STATUS] = "Model does not exist.";
                string dump = ret.dump();
		logger.error(" " + dump);
                res << dump;
                return ret;
            }   

	    auto relations = req_json["relations"];

	    // Edges not found
            if(relations.empty()) {
                ret[ms.STATUS] = "Edges not found in input";
                string dump = ret.dump();
		logger.error(" " + dump);
                res << dump;
                return ret;
            }

            // See if this model is available for training
	    bool busy = model_status_json[ms.BUSY];
	    if(busy) {
                ret[ms.PROGRESS] = model_status_json[ms.PROGRESS];
                ret[ms.STATUS] = "Model is busy(" 
                    + (string)model_status_json[ms.STATUS] 
		    + "), please wait until it finishes before editing.";
                string dump = ret.dump();
		logger.warning(" " + dump);
                res << dump;
                return ret;
            }  

	    // Get the model row from the database
            json query_result = sqlite3DB->select_delphimodel_row(modelId);

	    // deserialize the model
	    string modelString = query_result["model"];
            AnalysisGraph G;
            G = G.deserialize_from_json_string(modelString, false);

	    // freeze edges
	    for(auto relation : relations) {
		string source_name = relation["source"];
		string target_name = relation["target"];
		vector<double> weights = relation["weights"];
		double scaled_weight = weights.front();
		int polarity = relation["polarity"];

		// test the return value
		unsigned short retcode = G.freeze_edge_weight(
                    source_name,
                    target_name,
                    scaled_weight,
                    polarity
                );

		if (retcode) {
                    switch (retcode) {
                        case 1: ret[ms.STATUS] = "scaled_weight "
                            + to_string(scaled_weight)
                            + " outside accepted range";
                        break;
                        case 2: ret[ms.STATUS] = "Source concept '"
                            + source_name
                            + "' does not exist";
                        break;
                        case 4: ret[ms.STATUS] = "Target concept '"
                            + target_name
                            + "' does not exist";
                        break;
                        case 8: ret[ms.STATUS] = "There is no edge from '"
                            + source_name
                            + "' to '"
                            + target_name
                            + "'";
                        break;
                        default: 
                        break;
                    }
                    string dump = ret.dump();
		    logger.error(" " + dump);
                    res << dump;
                    return ret;
		}
	    }

	    // train model in another thread
	    size_t kde_kernels = Model::get_kde_kernels();
	    int burn = Model::get_burn();
	    int sampling_resolution = Model::get_sampling_resolution();

            G.set_n_kde_kernels(kde_kernels);

            if(!sqlite3DB->insert_into_delphimodel(
                modelId,
                G.serialize_to_json_string(false)
            )) {
                ret[ms.STATUS] = "server error: unable to insert into delphi model";
                string dump = ret.dump();
		logger.error(" " + dump);
                res << dump;
                return ret;
	    }

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
                ret["status"] = "server error: edit-edges training";
                string dump = ret.dump();
		logger.error(" " + dump);
                res << dump;
                return ret;
            }
	   
	    // report success 
	    ret[ms.STATUS] = "Edges frozen: "
              + to_string(relations.size())
              + ". Model is in training.";
            string dump = ret.dump();
            logger.info(" " + dump);
            res << dump;
            return ret;
        });


    /* openApi 3.0.0
     * Get the status of an existing model
     */
    mux.handle("/models/{modelId}")
        .get([&sqlite3DB](served::response& res, const served::request& req) {

            string modelId = req.params["modelId"];
            ModelStatus ms(modelId, sqlite3DB);
            json ret;
            ret[ms.MODEL_ID] = modelId;
            Logger logger;
            logger.info("");
            logger.info("### DELPHI ENDPOINT: models/" + modelId);

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
                ret["status"] = "Invalid model ID";
                string dump = ret.dump();
                logger.error(" " + dump);
                res << dump;
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

                string dump = response.dump();
                logger.info(" " + dump);
                res << dump;
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
