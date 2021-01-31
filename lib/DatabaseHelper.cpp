#include "DatabaseHelper.hpp"


using namespace std;

// DB class
//conn to sqlite
// methods to connect, CRUD, save model, 
// serialize
// takes exp id gives result

Database::Database(){
	int rc = sqlite3_open(getenv("DELPHI_DB"), &db);

	if (rc == 1)
  	  	throw "Could not open db\n";
}

Database::~Database(){
	sqlite3_close(db);
}
  	
int Database::prepareStatement(const char* query, sqlite3_stmt* stmt){
	return sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
}

vector<string> Database::Database_Create(string database_name){

}

vector<string> Database::Database_Read(string query){
  	vector<string> matches;
  	sqlite3_stmt* stmt = nullptr;
  	int rc = prepareStatement(query.c_str(), stmt);
   	while (sqlite3_step(stmt) == SQLITE_ROW) {
   	  matches.push_back(
   	      string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))));
   	}
   	sqlite3_finalize(stmt);
   	stmt = nullptr;
   	return matches;
}


vector<string> Database::Database_Write(string database_name){

}


vector<string> Database::Database_Delete(string database_name){

}


