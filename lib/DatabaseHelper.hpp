#pragma once

#include <nlohmann/json.hpp>
#include <range/v3/all.hpp>
#include <sqlite3.h>
#include <vector>

class Database {
  private:
  sqlite3* db;

  public:
  Database();

  ~Database();

  std::vector<std::string> read_column_text(std::string query);

  std::vector<std::string> read_column_text_query(std::string table_name,
                                                  std::string column_name);

  std::vector<std::string>
  read_column_text_query_where(std::string table_name,
                               std::string column_name,
                               std::string where_column_name,
                               std::string where_value);

  nlohmann::json select_row(
    std::string table,
    std::string id,
    std::string output_field
  );

  nlohmann::json select_delphimodel_row(std::string modelId);

  nlohmann::json select_causemosasyncexperimentresult_row(std::string modelId);

  bool insert(std::string insert_query);

  bool insert_into_delphimodel(std::string id, std::string model);

  bool insert_into_causemosasyncexperimentresult(std::string id,
                                                 std::string status,
                                                 std::string experimentType,
                                                 std::string results);

  bool exec_query(std::string query);
};
