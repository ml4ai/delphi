#include "DatabaseHelper.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>


using namespace std;
using json = nlohmann::json;

/*
    Database class constructor to open sqlite3 connection
*/
Database::Database() {
    int rc = sqlite3_open(getenv("DELPHI_DB"), &db);

    if (rc) {
        // Show an error message
        cout << "DB Error: " << sqlite3_errmsg(db) << endl;
        // Close the connection
        sqlite3_close(db);
        // Return an error
        throw "Could not open db\n";
    }
}

Database::~Database() {
    sqlite3_close(db);
    db = nullptr;
}

// Create a callback function
int callback(void* NotUsed, int argc, char** argv, char** azColName) {
    // Return successful
    return 0;
}

/*
    Query to select/read one column and return a vector of string filled with
   the column value of any table Query format: SELECT <column_name> from
   <table_name>; SELECT <column_name> from <table_name>  WHERE
   <where_column_name> = <where_value> ;
*/
vector<string> Database::read_column_text(string query) {
    vector<string> matches;
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
	const unsigned char* col_text = sqlite3_column_text(stmt, 0);
	if(col_text != nullptr) {
            string row_str = string(reinterpret_cast<const char*>(col_text));
            matches.push_back(row_str);
	} 
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
    return matches;
}

/*
    Select/read one column and all rows of any table
    Query format:
        SELECT <column_name> from <table_name>;

*/
vector<string> Database::read_column_text_query(string table_name,
                                                string column_name) {
    string query = "SELECT " + column_name + " from '" + table_name + "' ;";
    vector<string> matches = this->read_column_text(query);
    return matches;
}

/*
    Select/read one column and all rows of any table with where conditioned
   column and its value passed as parameters Query format: SELECT <column_name>
   from <table_name>  WHERE  <where_column_name> = <where_value> ;

*/
vector<string> Database::read_column_text_query_where(string table_name,
                                                      string column_name,
                                                      string where_column_name,
                                                      string where_value) {
    string query = "SELECT " + column_name + " from '" + table_name +
                   "'  WHERE " + where_column_name + " = '" + where_value +
                   "' ;";
    vector<string> matches = this->read_column_text(query);
    return matches;
}

/*
    Select the primary key and the model from the delphimodel table
*/
json Database::select_delphimodel_row(string modelId) {
    json matches;
    sqlite3_stmt* stmt = nullptr;
    string query =
        "SELECT * from delphimodel WHERE id='" + modelId + "'  LIMIT 1;";
    int rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        matches["id"] =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
        matches["model"] =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
    return matches;
}

/*
    Select the row in the table that matches the modelId
*/
json Database::select_row(string table, string modelId, string output_field) {
    json matches;
    sqlite3_stmt* stmt = nullptr;
    string query =
        "SELECT * from " + table + " WHERE id='" + modelId + "'  LIMIT 1;";
    int rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        matches["id"] =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
        matches[output_field] =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
    return matches;
}

/*
    Select/read all column and 1 rows of causemosasyncexperimentresult table
*/
json Database::select_causemosasyncexperimentresult_row(string modelId) {
    json matches;
    sqlite3_stmt* stmt = nullptr;
    string query = "SELECT * from causemosasyncexperimentresult WHERE id='" +
                   modelId + "' LIMIT 1;";
    int rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        matches["id"] =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
        matches["status"] =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
        matches["experimentType"] =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
        matches["results"] =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
    }

    sqlite3_finalize(stmt);
    stmt = nullptr;
    return matches;
}

/*
    Execute insert query string on any table
*/
bool Database::insert(string insert_query) {
    return exec_query(insert_query);
}

/*
    Execute insert/replace query string on delphimodel table for 1 row
*/
bool Database::insert_into_delphimodel(string id, string model) {
    string query =
        "INSERT OR REPLACE INTO delphimodel ('id', 'model') VALUES ('" + id +
        "', '" + model + "');";
    return insert(query);
}

/*
    Execute insert/replace query string on causemosasyncexperimentresult table
   for 1 row
*/
bool Database::insert_into_causemosasyncexperimentresult(string id,
                                                         string status,
                                                         string experimentType,
                                                         string results) {
    string query = "INSERT OR REPLACE INTO causemosasyncexperimentresult "
                   "('id', 'status', 'experimentType', 'results') VALUES ('" +
                   id + "', '" + status + "', '" + experimentType + "', '" +
                   results + "'); ";

    return insert(query);
}

/* Execute the query on the database, retry if busy, report errors */
bool Database::exec_query(string query) {

  int max_attempts = 100;  // usually only takes a few attempts.
  string fn = "DatabaseHelper::exec ";

  for(int i = 0; i < max_attempts; i ++) {
    int rc = sqlite3_exec(db, query.c_str(), callback, 0, NULL);
    switch (rc) {
      case SQLITE_OK:  // success
        if(i > 0) {
          cout << fn << "succeeded after " << i+1 << " attempts " << endl;
        }
        return true;
      case SQLITE_BUSY:  // blocked by busy db, wait a moment and retry
        this_thread::sleep_for(std::chrono::seconds(1));
        break;
      default:  // failure
        cout << fn << "failed with error code: " << rc << endl;
        return false;
    }
  }

  // maxed out the number of attempts, something is wrong.
  cout << fn << "could not execute after " << max_attempts << " attempts";
  return false;
}
