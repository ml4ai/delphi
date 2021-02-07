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
	int rc =  sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
	return rc;
}

// Create a callback function  
int callbackCreate(void *NotUsed, int argc, char **argv, char **azColName){
    // Return successful
    return 0;
}

int Database::Database_Create(string table_name){
	/*
		Example query:
			CREATE TABLE PEOPLE (
      		"ID INT PRIMARY KEY     NOT NULL,
      		"NAME           TEXT    NOT NULL);
	*/
	// Todo: 1 func for all table if not exist
	string create_table_query = "";
	if(table_name == "delphimodel"){
		create_table_query= "CREATE TABLE " + table_name + " (
							 id TEXT PRIMARY KEY,
							 model TEXT NOT NULL,
							 );";
	} else if(table_name == "experimentresult"){
		create_table_query= "CREATE TABLE " + table_name + " (
							 baseType TEXT NOT NULL,
							 id TEXT PRIMARY KEY,
							 );";
		// Todo: more args
	} else if(table_name == "causemosasyncexperimentresult"){
		create_table_query= "CREATE TABLE " + table_name + " (
							 id TEXT PRIMARY KEY,
							 status TEXT,
							 experimentType TEXT,
							 results TEXT, // todo: type???? okay as text => nhollman
							 FOREIGN KEY (id) 
      						 	REFERENCES experimentresult (id) 
      						 	   ON DELETE CASCADE 
      						 	   ON UPDATE NO ACTION
							 );";
		// Todo: more args
	} else return -1;
	 
	int rc = sqlite3_exec(db, create_table_query.c_str(), callbackCreate, 0, NULL);
	return rc;
}

vector<string> Database::Database_Read_ColumnText(string query){
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

vector<string> Database::Database_Read_Query(string query){

  	vector<string> matches = this->Database_Read(query);
   	return matches;
}


vector<string> Database::Database_Write(string database_name){

}

vector<string> Database::Database_Update(string database_name){

}

vector<string> Database::Database_Delete_Table(string database_name){

}

vector<string> Database::Database_Delete_Row(string database_name){

}


