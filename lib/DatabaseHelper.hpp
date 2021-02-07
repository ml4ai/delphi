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
	public:
		sqlite3  *db = nullptr;
  		
		Database();

		~Database();
  	
  	int prepareStatement(const char* query, sqlite3_stmt* stmt);

    int Database_Create(string create_table_query);

  	vector<string> Database_Read(string query);

    vector<string> Database_Write(string database_name);

    vector<string> Database_Update(string database_name);

    vector<string> Database_Delete(string database_name);

};