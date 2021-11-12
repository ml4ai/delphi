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
  stop_monitoring();
}

void TrainingStatus::scheduler()
{
  while(!ag->get_trained()){
    this_thread::sleep_for(std::chrono::seconds(1));
    if(pThread != nullptr) {
      write_to_db();
    }
  }
}

void TrainingStatus::start_monitoring(AnalysisGraph *ag){
  this->ag = ag;

  if(pThread == nullptr) {
    pThread = new thread(&TrainingStatus::scheduler, this);
  }
}

void TrainingStatus::stop_monitoring(){
    if (pThread != nullptr)
    {
        if(pThread->joinable()) {
            pThread->join();
	}
        delete pThread;
    }
    pThread = nullptr;
}

void TrainingStatus::write_to_db() {

  if (ag != nullptr) {
    string model_id = ag->id;
    if(!model_id.empty()) {
	  
      json status;
      status["id"] = model_id;
      status["progress"] = ag->get_training_progress();
      status["trained"] = ag->get_trained();
      status["stopped"] = ag->get_stopped();
      status["log_likelihood"] = ag->get_log_likelihood();
      status["log_likelihood_previous"] = ag->get_previous_log_likelihood();
      status["log_likelihood_map"] = ag->get_log_likelihood_MAP();

      string query = "REPLACE INTO " + table + " VALUES ('" + model_id  + "', '" + status.dump() +  "');";


      database->insert(query);
    }
  }
}

string TrainingStatus::read_from_db(string modelId) {
  return database->select_training_status_row(modelId).dump();
}

// create the table if we need it.
void TrainingStatus::init_db() {
  string query = "CREATE TABLE IF NOT EXISTS " + table + " (id TEXT PRIMARY KEY, status TEXT NOT NULL);";

  database->insert(query);
}
