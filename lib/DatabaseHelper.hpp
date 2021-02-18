#ifndef DATABASEHELPER_H
#define DATABASEHELPER_H


#include <sqlite3.h>
#include <vector>
#include <range/v3/all.hpp>


using namespace std;

// DB class
//conn to sqlite
// methods to connect, CRUD, save model, 
// serialize
// takes exp id gives result

class Database {
	private:
	
  	sqlite3  *db;
  		

  public:

		Database();

		~Database();
  	
  	//int prepareStatement(const char* query, sqlite3_stmt* stmt);

    //int callbackCreate(void *NotUsed, int argc, char **argv, char **azColName);

    void Database_Create();

  	vector<string> Database_Read_ColumnText(string query);

    vector<string> Database_Read_Query(string column_name, string table_name);

    void Database_Insert(string insert_query);

    void Database_Update(string table_name, string column_name, string value, string where_column_name, string where_value);

    void Database_Drop_Table(string database_name);

    void Database_Delete_Row(string database_name);

};

#endif /* DATABASEHELPER_H */
