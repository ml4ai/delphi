#include "AnalysisGraph.hpp"
#include "itertools.hpp"
#include <sqlite3.h>

using namespace std;
using namespace delphi::utils;

AdjectiveResponseMap
construct_adjective_response_map(size_t n_kernels = DEFAULT_N_SAMPLES) {
  sqlite3* db;
  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);

  if (rc == 1)
    throw "Could not open db\n";

  sqlite3_stmt* stmt;
  const char* query = "select * from gradableAdjectiveData";
  rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);

  AdjectiveResponseMap adjective_response_map;

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    string adjective =
        string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
    double response = sqlite3_column_double(stmt, 6);
    if (in(adjective_response_map, adjective)) {
      adjective_response_map[adjective] = {response};
    }
    else {
      adjective_response_map[adjective].push_back(response);
    }
  }

  for (auto& [k, v] : adjective_response_map) {
    v = KDE(v).resample(n_kernels);
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return adjective_response_map;
}

void AnalysisGraph::construct_beta_pdfs() {

  double sigma_X = 1.0;
  double sigma_Y = 1.0;
  AdjectiveResponseMap adjective_response_map = construct_adjective_response_map();
  vector<double> marginalized_responses;
  for (auto [adjective, responses] : adjective_response_map) {
    for (auto response : responses) {
      marginalized_responses.push_back(response);
    }
  }

  marginalized_responses =
      KDE(marginalized_responses).resample(DEFAULT_N_SAMPLES);

  for (auto e : this->edges()) {
    vector<double> all_thetas = {};

    for (Statement stmt : this->graph[e].evidence) {
      Event subject = stmt.subject;
      Event object = stmt.object;

      string subj_adjective = subject.adjective;
      string obj_adjective = object.adjective;

      auto subj_responses = lmap(
          [&](auto x) { return x * subject.polarity; },
          get(adjective_response_map, subj_adjective, marginalized_responses));

      auto obj_responses = lmap(
          [&](auto x) { return x * object.polarity; },
          get(adjective_response_map, obj_adjective, marginalized_responses));

      for (auto [x, y] : iter::product(subj_responses, obj_responses)) {
        all_thetas.push_back(atan2(sigma_Y * y, sigma_X * x));
      }
    }

    this->graph[e].kde = KDE(all_thetas);

    // Initialize the initial β for this edge
    // TODO: Decide the correct way to initialize this
    this->graph[e].beta = this->graph[e].kde.mu;
  }
}

