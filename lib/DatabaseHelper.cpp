#include "DatabaseHelper.hpp"
#include<iostream>
#include <nlohmann/json.hpp>


using namespace std;
using json = nlohmann::json;


/* 
    Database class constructor to open sqlite3 connection 
*/
Database::Database(){
	//int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
	int rc = sqlite3_open("/Users/aishwaya/Desktop/delphi/delphi_test.db", &db);

	if (rc){
		// Show an error message
        cout << "DB Error: " << sqlite3_errmsg(db) << endl;
        // Close the connection
        sqlite3_close(db);
        // Return an error
  	  	throw "Could not open db\n";
	}
}

Database::~Database(){
	sqlite3_close(db);
	db = nullptr;
}


// Create a callback function  
int callback(void *NotUsed, int argc, char **argv, char **azColName){
    // Return successful
    return 0;
}

void Database::Database_Create(){  // todo
	/*
		Example query:
			CREATE TABLE PEOPLE (
      		"ID INT PRIMARY KEY     NOT NULL,
      		"NAME           TEXT    NOT NULL);
	*/
	// Save any error messages
 /*  	char *zErrMsg = 0;

	//string create_table_query = "CREATE TABLE IF NOT EXISTS delphimodel ( \
	//								id VARCHAR NOT NULL, \
	//								model VARCHAR, \
	//								PRIMARY KEY (id) \
	//							 );";

	string create_table_query = "CREATE TABLE IF NOT EXISTS delphimodel ( id VARCHAR NOT NULL, model VARCHAR, PRIMARY KEY (id) );";

	cout << "\nBefore" << endl;
	int rc = sqlite3_exec(db, create_table_query.c_str(), callback, 0, &zErrMsg);
	string zErrMsgstr(zErrMsg);
	cout << "\n" << zErrMsgstr << endl;
*/

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
	 
	
/*
		
	CREATE TABLE causemosasyncexperimentresult ( id TEXT PRIMARY KEY, status TEXT, experimentType TEXT, results TEXT, FOREIGN KEY (id)  REFERENCES experimentresult (id) ON DELETE CASCADE  ON UPDATE NO ACTION );
	
*/
}



/* 
    Query to select/read one column and return a vector of string filled with the column value of any table
    Query format: 
    	SELECT <column_name> from <table_name>;
    	SELECT <column_name> from <table_name>  WHERE  <where_column_name> = <where_value> ;

*/ 
vector<string> Database::Database_Read_ColumnText(string query){
  	vector<string> matches;
  	sqlite3_stmt* stmt = nullptr;
  	cout << query << endl;
  	int rc =  sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
   	while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
   	  matches.push_back(
   	      string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))));
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
vector<string> Database::Database_Read_ColumnText_Query(string table_name, string column_name){
	string query = "SELECT "+ column_name +" from '"+ table_name +"' ;";
  	vector<string> matches = this->Database_Read_ColumnText(query);
   	return matches;
}

/* 
    Select/read one column and all rows of any table with where conditioned column and its value passed as parameters
    Query format: 
    	SELECT <column_name> from <table_name>  WHERE  <where_column_name> = <where_value> ;

*/
vector<string> Database::Database_Read_ColumnText_Query_Where(string table_name, string column_name, string where_column_name, string where_value){
	string query = "SELECT "+ column_name +" from '"+ table_name +"'  WHERE "+ where_column_name +" = '"+ where_value +"' ;";
  	vector<string> matches = this->Database_Read_ColumnText(query);
   	return matches;
}


/* 
    Select/read all column and 1 rows of delphimodel table 
*/
json Database::Database_Read_delphimodel(string modelId){
  	json matches;
  	sqlite3_stmt* stmt = nullptr;
  	string query = "SELECT * from delphimodel WHERE id='"+modelId+"'  LIMIT 1;";
  	int rc =  sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
   	while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
   		matches["id"] = string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
   		matches["model"] = string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
   	}
   	sqlite3_finalize(stmt);
   	stmt = nullptr;
   	return matches;
}


/* 
    Select/read all column and 1 rows of causemosasyncexperimentresult table 
*/
json Database::Database_Read_causemosasyncexperimentresult(string modelId){
  	json matches;
  	sqlite3_stmt* stmt = nullptr;
  	string query = "SELECT * from causemosasyncexperimentresult WHERE id='"+modelId+"' LIMIT 1;";
  	int rc =  sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
   	while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
   		matches["id"] = string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
   		matches["status"] = string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
   		matches["experimentType"] = string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
   		matches["results"] = string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
   	}

   	if (rc != SQLITE_DONE)
    	cout << " Step Error: " << rc << " " <<  SQLITE_DONE;

   	sqlite3_finalize(stmt);
   	stmt = nullptr;
   	return matches;
}

/* 
    Execute insert query string on any table
*/
void Database::Database_Insert(string insert_query){
	char *zErrMsg = 0;
	int rc = sqlite3_exec(db, insert_query.c_str(), callback, 0, &zErrMsg);
}


/* 
    Execute insert/replace query string on delphimodel table for 1 row
*/
void Database::Database_InsertInto_delphimodel(string id, string model){
	string query = "INSERT OR REPLACE INTO delphimodel ('id', 'model') VALUES ('"+ id +"', '"+ model +"');";
    this->Database_Insert(query);
    
}


/* 
    Execute insert/replace query string on causemosasyncexperimentresult table for 1 row
*/
void Database::Database_InsertInto_causemosasyncexperimentresult(string id, string status, string experimentType, string results){
	string query = "INSERT OR REPLACE INTO causemosasyncexperimentresult ('id', 'status', 'experimentType', 'results') VALUES ('"+ id +"', '"+ status +"', '"+ experimentType +"', '"+ results +"'); ";

	cout << query << endl;
    this->Database_Insert(query);
    
}

/* 
    Execute update query string on any table for 1 column with where condition
*/
void Database::Database_Update(string table_name, string column_name, string value, string where_column_name, string where_value){
	string update_table_query = "UPDATE "+ table_name +" SET "+ column_name +" = '"+ value +"' WHERE "+ where_column_name +" = '"+ where_value +"';";
	int rc = sqlite3_exec(db, update_table_query.c_str(), callback, 0, NULL);
}




/* 
    Execute update query string on any table for 1 column with where condition
*/
void Database::Database_Delete_Rows(string table_name, string where_column_name, string where_value){
	string delete_table_query = "DELETE FROM  "+ table_name +" WHERE "+ where_column_name +" = '"+ where_value +"';";
	int rc = sqlite3_exec(db, delete_table_query.c_str(), callback, 0, NULL);
}
