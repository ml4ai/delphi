#include "AnalysisGraph.hpp"
#include <range/v3/all.hpp>
#include "libpq-fe.h"

using namespace std;
using namespace delphi::utils;

AdjectiveResponseMap construct_adjective_response_map(
    mt19937 gen,
    uniform_real_distribution<double>& uni_dist,
    normal_distribution<double>& norm_dist,
    size_t n_kernels = DEFAULT_N_SAMPLES) {

  const char   *conninfo;
  PGconn       *conn;
  PGresult     *res;
  conninfo = "dbname = delphi";

  conn = PQconnectdb(conninfo);

  if (PQstatus(conn) != CONNECTION_OK) {
    throw runtime_error("Could not open db. Do you have the DELPHI_DB "
        "environment correctly set to point to the Delphi database?");
  }
  cout << "Connection successful!!!!!!!!!!!" << endl; // todo


  const char* query = "select * from gradableAdjectiveData";
  res = PQexec(conn, query.c_str());

  AdjectiveResponseMap adjective_response_map;

  if (PQresultStatus(res) == PGRES_COMMAND_OK) {
      for (int i = 0; i < PQntuples(res); i++)
      {
          string adjective =
              string(reinterpret_cast<const char*>(PQgetvalue(res, i, 2)));  // todo // 2 column same as in sqlite?
          double response = PQgetvalue(res, i, 6);  // todo // 6 column same as in sqlite?
          if (in(adjective_response_map, adjective)) {
            adjective_response_map[adjective] = {response};
          }
          else {
            adjective_response_map[adjective].push_back(response);
          }
      }
  }
  PQclear(res);

  for (auto& [k, v] : adjective_response_map) {
    v = KDE(v).resample(n_kernels, gen, uni_dist, norm_dist);
  }
  PQfinish(conn);
  return adjective_response_map;
}

/*
 ============================================================================
 Public: Construct Beta Pdfs
 ============================================================================
*/

void AnalysisGraph::construct_beta_pdfs() {

  // The choice of sigma_X and sigma_Y is somewhat arbitrary here - we need to
  // come up with a principled way to select this value, or infer it from data.
  double sigma_X = 1.0;
  double sigma_Y = 1.0;
  AdjectiveResponseMap adjective_response_map =
      construct_adjective_response_map(
          this->rand_num_generator, this->uni_dist, this->norm_dist);
  vector<double> marginalized_responses;
  for (auto [adjective, responses] : adjective_response_map) {
    for (auto response : responses) {
      marginalized_responses.push_back(response);
    }
  }

  marginalized_responses = KDE(marginalized_responses)
                               .resample(DEFAULT_N_SAMPLES,
                                         this->rand_num_generator,
                                         this->uni_dist,
                                         this->norm_dist);

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

      for (auto [x, y] : ranges::views::cartesian_product(subj_responses, obj_responses)) {
        all_thetas.push_back(atan2(sigma_Y * y, sigma_X * x));
      }
    }

    this->graph[e].kde = KDE(all_thetas);

    // Initialize the initial β for this edge
    // TODO: Decide the correct way to initialize this
    this->graph[e].beta = this->graph[e].kde.mu;
  }
}
