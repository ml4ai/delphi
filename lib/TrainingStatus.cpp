#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "DatabaseHelper.hpp"
#include "TrainingStatus.hpp"
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;


TrainingStatus::TrainingStatus(){
  database = new Database();
}

TrainingStatus::~TrainingStatus(){
  stop_updating_db();
}

/* Start the thread that posts the status to the datbase */
void TrainingStatus::scheduler()
{
  while(!ag->get_trained()){
    this_thread::sleep_for(std::chrono::seconds(1));
    if(pThread != nullptr) {
      update_db();
    }
  }
}

/* Begin posting training status updates to the database on a regular interval */
void TrainingStatus::start_updating_db(AnalysisGraph *ag){
  this->ag = ag;

  if(pThread == nullptr) {
    pThread = new thread(&TrainingStatus::scheduler, this);
  }
}

/* Stop posting training status updates to the database */
void TrainingStatus::stop_updating_db(){
    if (pThread != nullptr)
    {
        if(pThread->joinable()) {
            pThread->join();
	}
        delete pThread;
    }
    pThread = nullptr;
}

/* set the "stopped" field to true */
string TrainingStatus::stop_training(string modelId){
  json status;
  status["id"] = modelId;
  status["status"] = "Endpoint 'training-stop' not yet implemented.";
  return status.dump();
}

/* write out the status as a string for the database */
json TrainingStatus::compose_status() {
  json status;
  if (ag != nullptr) {
    string model_id = ag->id;
    if(!model_id.empty()) {
      status["id"] = model_id;
      status["progress"] = ag->get_training_progress();
      status["trained"] = ag->get_trained();
      status["stopped"] = ag->get_stopped();
      status["log_likelihood"] = ag->get_log_likelihood();
      status["log_likelihood_previous"] = ag->get_previous_log_likelihood();
      status["log_likelihood_map"] = ag->get_log_likelihood_MAP();
    }
  }
  return status;
}

void TrainingStatus::update_db() {
  json status = compose_status();
  write_to_db(status);
}

/* write the model training status to the database */
void TrainingStatus::write_to_db(json status) {
  if(!status.empty()) {
    string id = status.value("id", "");
    string dump = status.dump();
    string query 
      = "REPLACE INTO " + table + " VALUES ('" + id + "', '" + dump +  "');";
    database->insert(query);
  }
}

/* Read the model training status from the database */
string TrainingStatus::read_from_db(string modelId) {
  return database->select_training_status_row(modelId).dump();
}

/* create the table if we need it.  */
void TrainingStatus::init_db() {
  string query = "CREATE TABLE IF NOT EXISTS " 
    + table 
    + " (id TEXT PRIMARY KEY, status TEXT NOT NULL);";

  database->insert(query);
}
