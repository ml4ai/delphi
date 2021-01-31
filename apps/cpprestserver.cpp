#include <iostream>
#include <served/served.hpp>
#include <nlohmann/json.hpp>
//#include "dbg"
#include "AnalysisGraph.hpp"

using namespace std;





// Todo: Database call handling
class DelphiModel(){
    public:
        string __tablename__;
        string id;
        string model;
        //id = db.Column(db.String, primary_key=True)
        //model = db.Column(db.String)

        DelphiModel(string _id, string _model){
            this->__tablename__ = "delphimodel";
            this->id = _id;
            this->model = _model;
        }        
};




class CauseMosAsyncExperimentResult(ExperimentResult){
    /* Placeholder docstring for class CauseMosAsyncExperimentResult. */
    public:
        string __tablename__;
        string id;
        string status;
        string experimentType;
        string results;
        unordered_map<string, string> __mapper_args__;


        CauseMosAsyncExperimentResult(){
            this->__tablename__ = "causemosasyncexperimentresult";
            //id = db.Column(
            //    db.String,
            //    db.ForeignKey("experimentresult.id"),
            //    primary_key=True,
            //    default=str(uuid4()),
            //)
            //status = db.Column(db.String, nullable=True)
            //experimentType = db.Column(db.String, nullable=True)
            //results = db.Column(JsonEncodedDict, nullable=True)
            __mapper_args__["polymorphic_identity"] = "CauseMosAsyncExperimentResult";
        }
};


int main(int argc, const char *argv[])
{

    served::multiplexer mux;
    mux.handle("/delphi/create-model")
        .post([](served::response & res, const served::request & req) {
            auto json_data = nlohmann::json::parse(req.body());
            //std::cout << "POST req: " << json_data << std::endl;
            size_t result = 1000;
            //const char* env_p = ;
            string ci(getenv("CI"));
            size_t DELPHI_N_SAMPLES((size_t)stoul(getenv("DELPHI_N_SAMPLES")));
            if( ci == "true") 
                result = 5;
            else if (DELPHI_N_SAMPLES) 
                result = DELPHI_N_SAMPLES;

            AnalysisGraph G = AnalysisGraph::from_causemos_json_string(json_data, result);

            //string id = "123";
            //string model = "abc";
            DelphiModel model(json_data["id"], G.serialize_to_json_string(false));
            // Todo: Database call handling
            //db.session.merge(model)
            //db.session.commit()

            res <<  nlohmann::json::parse(G.generate_create_model_response());
            //res << json_data.dump();
        });







    std::cout << "Try this example with:" << std::endl;
    std::cout << "curl -X POST \"http://localhost:8123/delphi/create-model\" -d @test.json --header \"Content-Type: application/json\" " << std::endl;

    served::net::server server("127.0.0.1", "8123", mux);
    server.run(10);

    return (EXIT_SUCCESS);
}





void runProjectionExperiment(const served::request & request, string modelID, string experiment_id, AnalysisGraph G, bool trained){
    auto request_body = nlohmann::json::parse(request.body());


    string startTime = request_body["experimentParam"]["startTime"];
    string endTime = request_body["experimentParam"]["endTime"];
    int numTimesteps = request_body["experimentParam"]["numTimesteps"];

    FormattedProjectionResult causemos_experiment_result = G.run_causemos_projection_experiment(
        request_body // todo: ??????
    )

    DelphiModel model;
    if(not trained){
        model = DelphiModel(modelID, G.serialize_to_json_string(false));

        // Todo: db
        //db.session.merge(model)
        //db.session.commit()
    }

    // todo ??????
    result = CauseMosAsyncExperimentResult.query.filter_by(
        id=experiment_id
    ).first()


    vector<vector<vector<double>>> formattedProjectionTimestepSample;
    for (const auto & [ key, value ] : causemos_experiment_result) {
        formattedProjectionTimestepSample.push_back(value);
    }
    if(formattedProjectionTimestepSample[0].size() < numTimesteps)
        result.status = "failed";
    else{
        result.status = "completed";

        // todo !!!!!1
        timesteps_nparr = np.round(
            np.linspace(startTime, endTime, numTimesteps)l
        )

        // The calculation of the 95% confidence interval about the median is
        // taken from:
        // https://www.ucl.ac.uk/child-health/short-courses-events/ \
        //     about-statistical-courses/research-methods-and-statistics/chapter-8-content-8
        int n = G.get_res();

        int lower_rank = (int)((n - 1.96 * sqrt(n)) / 2);
        int upper_rank = (int)((2 + n + 1.96 * sqrt(n)) / 2);

        lower_rank = lower_rank < 0 ? 0 : lower_rank;
        upper_rank = upper_rank >= n ? n - 1 : upper_rank;
        unordered_map<string, vector<string>> res_data; // vector<string> datatype ???
        res_data["data"] = {};
        result.results = res_data;
    }

}

 