#ifndef DATABASEHELPER_H
#define DATABASEHELPER_H


#include <sqlite3.h>
#include <vector>
#include <range/v3/all.hpp>
#include <nlohmann/json.hpp>


class Database {
  private:
  
    sqlite3  *db;
      

  public:

    Database();

    ~Database();
    
    void Database_Create();

    std::vector<std::string> Database_Read_ColumnText(std::string query);

    std::vector<std::string> Database_Read_ColumnText_Query(std::string table_name, std::string column_name);

    std::vector<std::string> Database_Read_ColumnText_Query_Where(std::string table_name, std::string column_name, std::string where_column_name, std::string where_value);

    nlohmann::json Database_Read_delphimodel(std::string modelId);

    nlohmann::json Database_Read_causemosasyncexperimentresult(std::string modelId);

    void Database_Insert(std::string insert_query);

    void Database_InsertInto_delphimodel(std::string id, std::string model);

    void Database_InsertInto_causemosasyncexperimentresult(std::string id, std::string status, std::string experimentType, std::string results);

    void Database_Update(std::string table_name, std::string column_name, std::string value, std::string where_column_name, std::string where_value);

    void Database_Delete_Rows(std::string table_name, std::string where_column_name, std::string where_value);

};

#endif /* DATABASEHELPER_H */