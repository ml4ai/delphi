#include "DatabaseHelper.hpp"
#include<iostream>

using namespace std;

// DB class
//conn to sqlite
// methods to connect, CRUD, save model, 
// serialize
// takes exp id gives result

Database::Database(){
	//int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
	int rc = sqlite3_open("/Users/aishwaya/Desktop/delphi/delphi_test.db", &db);

	if (rc){
		// Show an error message
        cout << "DB Error: " << sqlite3_errmsg(db) << endl;
        // Close the connection
        sqlite3_close(db);
        // Return an error
        //return(1);
  	  	throw "Could not open db\n";
	}
}

Database::~Database(){
	sqlite3_close(db);
	db = nullptr;
}
  	
//int Database::prepareStatement(const char* query, sqlite3_stmt* stmt){
//	int rc =  sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
//	return rc;
//}

// Create a callback function  
int callbackCreate(void *NotUsed, int argc, char **argv, char **azColName){
    // Return successful
    return 0;
}

void Database::Database_Create(){
	/*
		Example query:
			CREATE TABLE PEOPLE (
      		"ID INT PRIMARY KEY     NOT NULL,
      		"NAME           TEXT    NOT NULL);
	*/
	// Save any error messages
   	char *zErrMsg = 0;

	//string create_table_query = "CREATE TABLE IF NOT EXISTS delphimodel ( \
	//								id VARCHAR NOT NULL, \
	//								model VARCHAR, \
	//								PRIMARY KEY (id) \
	//							 );";

	string create_table_query = "CREATE TABLE IF NOT EXISTS delphimodel ( id VARCHAR NOT NULL, model VARCHAR, PRIMARY KEY (id) );";

	cout << "\nBefore" << endl;
	int rc = sqlite3_exec(db, create_table_query.c_str(), callbackCreate, 0, &zErrMsg);
	string zErrMsgstr(zErrMsg);
	cout << "\n" << zErrMsgstr << endl;

	//create_table_query= "CREATE TABLE experimentresult (
	//						\"baseType\" VARCHAR,
	//						id VARCHAR NOT NULL,
	//						PRIMARY KEY (id)
	//					 );";
	

	//	// Todo: more args
	//} else if(table_name == "causemosasyncexperimentresult"){
	//	create_table_query= "CREATE TABLE " + table_name + " (
	//						 id TEXT PRIMARY KEY,
	//						 status TEXT,
	//						 experimentType TEXT,
	//						 results TEXT, // todo: type???? okay as text => nhollman
	//						 FOREIGN KEY (id) 
    //  						 	REFERENCES experimentresult (id) 
    //  						 	   ON DELETE CASCADE 
    //  						 	   ON UPDATE NO ACTION
	//						 );";
	//
	 
	


}


vector<string> Database::Database_Read_ColumnText(string query){
  	vector<string> matches;
  	sqlite3_stmt* stmt = nullptr;
  	//const char **tail;
  	cout << query << endl;
  	int rc =  sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
  	//cout << SQLITE_ROW << endl;
   	while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
   	  matches.push_back(
   	      string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))));
   	}
   	sqlite3_finalize(stmt);
   	stmt = nullptr;
   	return matches;
}

vector<string> Database::Database_Read_Query(string column_name, string table_name){
	string query = "SELECT "+ column_name +" from '"+ table_name +"' ;";
  	vector<string> matches = this->Database_Read_ColumnText(query);
   	return matches;
}


void Database::Database_Insert(string insert_query){
	char *zErrMsg = 0;
	int rc = sqlite3_exec(db, insert_query.c_str(), callbackCreate, 0, &zErrMsg);
}


void Database::Database_InsertInto_delphimodel(string id, string model){
	string query = "INSERT INTO delphimodel ('id', 'model') VALUES ('"+ id +"', '"+ model +"');";
    this->Database_Insert(query);
    
    //query = "SELECT Source from concept_to_indicator_mapping WHERE Indicator = 'TEST';";
    //vector<string> matches = sqlite3DB->Database_Read_ColumnText(query);
}

void Database::Database_Update(string table_name, string column_name, string value, string where_column_name, string where_value){
	string update_table_query = "UPDATE "+ table_name +" SET "+ column_name +" = '"+ value +"' WHERE "+ where_column_name +" = '"+ where_value +"';";
	int rc = sqlite3_exec(db, update_table_query.c_str(), callbackCreate, 0, NULL);
}

