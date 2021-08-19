#include <sqlite3.h>
#include "AnalysisGraph.hpp"

using namespace std;
using namespace delphi::utils;

void AnalysisGraph::write_model_to_db(string model_id) {
  if (!model_id.empty()) {
    sqlite3* db = nullptr;
    int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
    //int rc = sqlite3_open("/tmp/test.db", &db);

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

/**
 * This is a helper function used by construct_theta_pdfs()
 */
AdjectiveResponseMap AnalysisGraph::construct_adjective_response_map(
    mt19937 gen,
    uniform_real_distribution<double>& uni_dist,
    normal_distribution<double>& norm_dist,
    size_t n_kernels
) {
  sqlite3* db = nullptr;
  sqlite3_open(getenv("DELPHI_DB"), &db);
  int rc = sqlite3_errcode(db);

  if (rc != SQLITE_OK)
    throw "Could not open db\n";

  sqlite3_stmt* stmt = nullptr;
  const char* query = "select * from gradableAdjectiveData";
  rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);

  AdjectiveResponseMap adjective_response_map;

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    string adjective =
        string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
    double response = sqlite3_column_double(stmt, 6);
    if (!in(adjective_response_map, adjective)) {
      adjective_response_map[adjective] = {response};
    }
    else {
      adjective_response_map[adjective].push_back(response);
    }
  }

  for (auto& [k, v] : adjective_response_map) {
    v = KDE(v).resample(n_kernels, gen, uni_dist, norm_dist);
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  stmt = nullptr;
  db = nullptr;
  return adjective_response_map;
}
