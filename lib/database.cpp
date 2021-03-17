#include <sqlite3.h>
#include "AnalysisGraph.hpp"

using namespace std;

void AnalysisGraph::write_model_to_db(string model_id) {
  if (!model_id.empty()) {
    sqlite3* db = nullptr;
    // int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
    int rc = sqlite3_open("/tmp/test.db", &db);

    if (rc != SQLITE_OK) {
      cout << "Could not open db\n";
      // throw "Could not open db\n";
    }

    char* zErrMsg = 0;
    string query = "replace into delphimodel values ('" + model_id + "', '" +
                   this->serialize_to_json_string(false) + "');";
    rc = sqlite3_exec(db, query.c_str(), NULL, NULL, &zErrMsg);

    if (rc != SQLITE_OK) {
      cout << "Could not write\n";
      cout << zErrMsg << endl;
    }

    sqlite3_close(db);
    db = nullptr;
  }
}