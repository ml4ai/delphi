#include "AnalysisGraph.hpp"
#include "data.hpp"
#include "spdlog/spdlog.h"
#include <boost/range/algorithm/for_each.hpp>
#include <range/v3/all.hpp>
#include <sqlite3.h>

using namespace std;
using namespace fmt::literals;
using namespace delphi::utils;
using spdlog::debug, spdlog::error, spdlog::info;

using boost::for_each, boost::graph_traits, boost::clear_vertex,
    boost::remove_vertex;
using Eigen::VectorXd;
using fmt::print, fmt::format;
using spdlog::debug;
using spdlog::error;
using spdlog::warn;

// Forward declarations
class Node;
class Indicator;


/*
 ============================================================================
 Private: Utilities
 ============================================================================
*/

void AnalysisGraph::clear_state() {
  this->A_beta_factors.clear();

  // Clear the multimap that keeps track of cells in the transition
  // matrix that are dependent on each β.
  this->beta2cell.clear();

  // Clear the set of all the β dependent cells
  this->beta_dependent_cells.clear();
}

void AnalysisGraph::initialize_random_number_generator() {
  // Define the random number generator
  // All the places we need random numbers share this generator

  this->rand_num_generator = RNG::rng()->get_RNG();

  // Uniform distribution used by the MCMC sampler
  this->uni_dist = uniform_real_distribution<double>(0.0, 1.0);

  // Normal distrubution used to perturb β
  this->norm_dist = normal_distribution<double>(0.0, 1.0);
}

void AnalysisGraph::remove_node(int node_id) {
  // Delete all the edges incident to this node
  clear_vertex(node_id, this->graph);

  // Remove the vertex
  remove_vertex(node_id, this->graph);

  // Update the internal meta-data
  for (int vert_id : this->node_indices()) {
    this->name_to_vertex.at((*this)[vert_id].name) = vert_id;
  }
}

void AnalysisGraph::allocate_A_beta_factors() {
  this->A_beta_factors.clear();

  int num_verts = this->num_vertices();

  for (int vert = 0; vert < num_verts; ++vert) {
    this->A_beta_factors.push_back(
        vector<shared_ptr<Tran_Mat_Cell>>(num_verts));
  }
}

void AnalysisGraph::find_all_paths_between(int start,
                                           int end,
                                           int cutoff = -1) {
  // Mark all the vertices are not visited
  for_each(this->node_indices(), [&](int v) { (*this)[v].visited = false; });

  // Create a vector of ints to store paths.
  vector<int> path;

  this->find_all_paths_between_util(start, end, path, cutoff);
}

void AnalysisGraph::find_all_paths_between_util(int start,
                                                int end,
                                                vector<int>& path,
                                                int cutoff) {
  // Mark the current vertex visited
  (*this)[start].visited = true;

  // Add this vertex to the path
  path.push_back(start);

  // If current vertex is the destination vertex, then
  //   we have found one path.
  //   Add this cell to the Tran_Mat_Object that is tracking
  //   this transition matrix cell.
  if (start == end) {
    // Add this path to the relevant transition matrix cell
    if (!A_beta_factors[path.back()][path[0]]) {
      this->A_beta_factors[path.back()][path[0]].reset(
          new Tran_Mat_Cell(path[0], path.back()));
    }

    this->A_beta_factors[path.back()][path[0]]->add_path(path);

    // This transition matrix cell is dependent upon Each β along this path.
    pair<int, int> this_cell = make_pair(path.back(), path[0]);

    beta_dependent_cells.insert(this_cell);

    for (int v = 0; v < path.size() - 1; v++) {
      this->beta2cell.insert(
          make_pair(make_pair(path[v], path[v + 1]), this_cell));
    }
  }
  else if (cutoff != 0) {
    cutoff--;
    // Current vertex is not the destination
    // Recursively process all the vertices adjacent to the current vertex
    for_each(this->successors(start), [&](int v) {
      if (!(*this)[v].visited) {
        this->find_all_paths_between_util(v, end, path, cutoff);
      }
    });
  }

  // Remove current vertex from the path and make it unvisited
  path.pop_back();
  (*this)[start].visited = false;
}

int AnalysisGraph::calculate_num_timesteps(int start_year,
                                           int start_month,
                                           int end_year,
                                           int end_month) {
  assert(start_year <= end_year);

  if (start_year == end_year) {
    assert(start_month <= end_month);
  }

  int diff_year = end_year - start_year;
  int year_to_month = diff_year * 12;

  // NOTE: I am adding 1 here itself so that I do not have to add it outside
  return year_to_month - (start_month - 1) + (end_month - 1) + 1;
}


/*
 ============================================================================
 Private: Accessors
 ============================================================================
*/

int AnalysisGraph::get_vertex_id(string concept) {
  try {
    return this->name_to_vertex.at(concept);
  }
  catch (const out_of_range& oor) {
    throw out_of_range("Concept \"{}\" not in CAG!"_format(concept));
  }
}

vector<Node> AnalysisGraph::get_successor_list(string node) {
  vector<Node> successors = {};
  for (int successor : this->successors(node)) {
    successors.push_back((*this)[successor]);
  }
  return successors;
}

vector<Node> AnalysisGraph::get_predecessor_list(string node) {
  vector<Node> predecessors = {};
  for (int predecessor : this->predecessors(node)) {
    predecessors.push_back((*this)[predecessor]);
  }
  return predecessors;
}


/*
 ============================================================================
 Private: Get Training Data Sequence
 ============================================================================
*/

void AnalysisGraph::set_observed_state_sequence_from_data(int start_year,
                                                          int start_month,
                                                          int end_year,
                                                          int end_month,
                                                          string country,
                                                          string state,
                                                          string county) {
  this->observed_state_sequence.clear();

  // Access
  // [ timestep ][ vertex ][ indicator ]
  this->observed_state_sequence = ObservedStateSequence(this->n_timesteps);

  int year = start_year;
  int month = start_month;

  for (int ts = 0; ts < this->n_timesteps; ts++) {
    this->observed_state_sequence[ts] =
        get_observed_state_from_data(year, month, country, state, county);

    if (month == 12) {
      year++;
      month = 1;
    }
    else {
      month++;
    }
  }
}

vector<vector<vector<double>>> AnalysisGraph::get_observed_state_from_data(
    int year, int month, string country, string state, string county) {
  using ranges::to;
  using ranges::views::transform;

  int num_verts = this->num_vertices();

  // Access
  // [ vertex ][ indicator ]
  vector<vector<vector<double>>> observed_state(num_verts);

  for (int v = 0; v < num_verts; v++) {
    vector<Indicator>& indicators = (*this)[v].indicators;

    for (auto& ind : indicators) {
      auto vals = get_data_value(ind.get_name(),
                                 country,
                                 state,
                                 county,
                                 year,
                                 month,
                                 ind.get_unit(),
                                 this->data_heuristic);

      observed_state[v].push_back(vals);
    }
  }

  return observed_state;
}


/*
 ============================================================================
 Private: Syntheitc Data Experiment
 ============================================================================
*/

void AnalysisGraph::init_betas_to(InitialBeta ib) {
  switch (ib) {
  // Initialize the initial β for this edge
  // Note: I am repeating the loop within each case for efficiency.
  // If we embed the switch withn the for loop, there will be less code
  // but we will evaluate the switch for each iteration through the loop
  case InitialBeta::ZERO:
    for (EdgeDescriptor e : this->edges()) {
      graph[e].beta = 0;
    }
    break;
  case InitialBeta::ONE:
    for (EdgeDescriptor e : this->edges()) {
      graph[e].beta = 1.0;
    }
    break;
  case InitialBeta::HALF:
    for (EdgeDescriptor e : this->edges()) {
      graph[e].beta = 0.5;
    }
    break;
  case InitialBeta::MEAN:
    for (EdgeDescriptor e : this->edges()) {
      graph[e].beta = graph[e].kde.mu;
    }
    break;
  case InitialBeta::RANDOM:
    for (EdgeDescriptor e : this->edges()) {
      // this->uni_dist() gives a random number in range [0, 1]
      // Multiplying by 2 scales the range to [0, 2]
      // Sustracting 1 moves the range to [-1, 1]
      graph[e].beta = this->uni_dist(this->rand_num_generator) * 2 - 1;
    }
    break;
  }
}

void AnalysisGraph::set_random_initial_latent_state() {
  int num_verts = this->num_vertices();

  this->set_default_initial_state();

  for (int v = 0; v < num_verts; v++) {
    this->s0(2 * v + 1) = 0.1 * this->uni_dist(this->rand_num_generator);
  }
}

void AnalysisGraph::generate_synthetic_latent_state_sequence() {
  int num_verts = this->num_vertices();

  // Allocate memory for synthetic_latent_state_sequence
  this->synthetic_latent_state_sequence.clear();
  this->synthetic_latent_state_sequence =
      vector<VectorXd>(this->n_timesteps, VectorXd(num_verts * 2));

  this->synthetic_latent_state_sequence[0] = this->s0;

  for (int ts = 1; ts < this->n_timesteps; ts++) {
    this->synthetic_latent_state_sequence[ts] =
        this->A_original * this->synthetic_latent_state_sequence[ts - 1];
  }
}

void AnalysisGraph::
    generate_synthetic_observed_state_sequence_from_synthetic_latent_state_sequence() {
  using ranges::to;
  using ranges::views::transform;
  // Allocate memory for observed_state_sequences
  this->test_observed_state_sequence.clear();
  this->test_observed_state_sequence = PredictedObservedStateSequence(
      this->n_timesteps, vector<vector<double>>());

  this->test_observed_state_sequence =
      this->synthetic_latent_state_sequence |
      transform([this](VectorXd latent_state) {
        return this->sample_observed_state(latent_state);
      }) |
      to<vector>();
}

vector<vector<double>>
AnalysisGraph::sample_observed_state(VectorXd latent_state) {
  using ranges::to;
  using ranges::views::transform;
  int num_verts = this->num_vertices();

  assert(num_verts == latent_state.size() / 2);

  vector<vector<double>> observed_state(num_verts);

  for (int v = 0; v < num_verts; v++) {
    vector<Indicator>& indicators = (*this)[v].indicators;

    observed_state[v] = vector<double>(indicators.size());

    // Sample observed value of each indicator around the mean of the
    // indicator
    // scaled by the value of the latent state that caused this observation.
    // TODO: Question - Is ind.mean * latent_state[ 2*v ] correct?
    //                  Shouldn't it be ind.mean + latent_state[ 2*v ]?
    observed_state[v] = indicators | transform([&](Indicator ind) {
                          normal_distribution<double> gaussian(
                              ind.mean * latent_state[2 * v], ind.stdev);

                          return gaussian(this->rand_num_generator);
                        }) |
                        to<vector>();
  }

  return observed_state;
}


/*
 ============================================================================
 Public: Graph building and modification
 ============================================================================
*/

void AnalysisGraph::add_node(string concept) {
  if (!in(this->name_to_vertex, concept)) {
    int v = boost::add_vertex(this->graph);
    this->name_to_vertex[concept] = v;
    (*this)[v].name = concept;
  }
  else {
    debug("AnalysisGraph::add_node()\n\tconcept {} already exists!\n", concept);
  }
}

void AnalysisGraph::add_edge(CausalFragment causal_fragment) {
  Event subject = Event(causal_fragment.first);
  Event object = Event(causal_fragment.second);

  string subj_name = subject.concept_name;
  string obj_name = object.concept_name;

  if (subj_name.compare(obj_name) != 0) { // Guard against self loops
    // Add the nodes to the graph if they are not in it already
    this->add_node(subj_name);
    this->add_node(obj_name);

    // Add the edge to the graph if it is not in it already
    auto [e, exists] = boost::add_edge(this->name_to_vertex[subj_name],
                                       this->name_to_vertex[obj_name],
                                       this->graph);

    this->graph[e].evidence.push_back(Statement{subject, object});
  }
  else {
    debug("AnalysisGraph::add_edge\n"
          "\tWARNING: Prevented adding a self loop for the concept {}",
          subj_name);
  }
}

pair<EdgeDescriptor, bool> AnalysisGraph::add_edge(int source, int target) {
  return boost::add_edge(source, target, this->graph);
}

pair<EdgeDescriptor, bool> AnalysisGraph::add_edge(int source, string target) {
  return boost::add_edge(source, this->get_vertex_id(target), this->graph);
}

pair<EdgeDescriptor, bool> AnalysisGraph::add_edge(string source, int target) {
  return boost::add_edge(this->get_vertex_id(source), target, this->graph);
}

pair<EdgeDescriptor, bool> AnalysisGraph::add_edge(string source,
                                                   string target) {
  return boost::add_edge(this->get_vertex_id(source),
                         this->get_vertex_id(target),
                         this->graph);
}

/*
 * The refactoring of the remove_node() method was buggy.
 * It caused AnalysisGraph to crash.
 * I replaced it with the previous implementation
*/
void AnalysisGraph::remove_node(string concept) {
    auto node_to_remove = this->name_to_vertex.extract(concept);

      if (node_to_remove) // Concept is in the CAG
      {
        // Note: This is an overlaoded private method that takes in a vertex id
        this->remove_node(node_to_remove.mapped());
      }
      else // The concept is not in the graph
      {
        throw out_of_range("Concept \"{}\" not in CAG!"_format(concept));
      }
}

void AnalysisGraph::remove_nodes(unordered_set<string> concepts) {
  vector<string> invalid_concepts;

  for (string concept : concepts) {
    auto node_to_remove = this->name_to_vertex.extract(concept);

    // Concept is in the CAG
    if (node_to_remove) {
      // Note: This is an overlaoded private method that takes in a vertex id
      this->remove_node(node_to_remove.mapped());
    }
    else // Concept is not in the CAG
    {
      invalid_concepts.push_back(concept);
    }
  }

  if (invalid_concepts.size() > 0) {
    // There were some invalid concepts
    error("AnalysisGraph::remove_nodes()\n"
          "\tThe following concepts were not present in the CAG!\n");
    for (string invalid_concept : invalid_concepts) {
      cerr << "\t\t" << invalid_concept << endl;
    }
  }
}

void AnalysisGraph::remove_edge(string src, string tgt) {
  // Remove the edge
  boost::remove_edge(
      this->get_vertex_id(src), this->get_vertex_id(tgt), this->graph);
}

void AnalysisGraph::remove_edges(vector<pair<string, string>> edges) {
  vector<pair<int, int>> edge_ids = vector<pair<int, int>>(edges.size());

  set<string> invalid_sources;
  set<string> invalid_targets;
  set<pair<string, string>> invalid_edges;

  std::transform(edges.begin(),
                 edges.end(),
                 edge_ids.begin(),
                 [this](pair<string, string> edge) {
                   int src_id;
                   int tgt_id;

                   // Flag invalid source vertices
                   try {
                     src_id = this->get_vertex_id(edge.first);
                   }
                   catch (const out_of_range& oor) {
                     src_id = -1;
                   }

                   // Flag invalid target vertices
                   try {
                     tgt_id = this->get_vertex_id(edge.second);
                   }
                   catch (const out_of_range& oor) {
                     tgt_id = -1;
                   }

                   // Flag invalid edges
                   if (src_id != -1 && tgt_id != -1) {
                     pair<int, int> edge_id = make_pair(src_id, tgt_id);

                     if (in(this->beta2cell, edge_id)) {
                       src_id = -2;
                     }
                   }

                   return make_pair(src_id, tgt_id);
                 });

  bool has_invalid_sources = false;
  bool has_invalid_targets = false;
  bool has_invalid_edges = false;

  for (int e = 0; e < edge_ids.size(); e++) {
    bool valid_edge = true;

    if (edge_ids[e].first == -1) {
      invalid_sources.insert(edges[e].first);
      valid_edge = false;
      has_invalid_sources = true;
    }

    if (edge_ids[e].second == -1) {
      invalid_targets.insert(edges[e].second);
      valid_edge = false;
      has_invalid_targets = true;
    }

    if (edge_ids[e].first == -2) {
      invalid_edges.insert(edges[e]);
      valid_edge = false;
      has_invalid_edges = true;
    }

    if (valid_edge) {
      // Remove the edge
      boost::remove_edge(edge_ids[e].first, edge_ids[e].second, this->graph);
    }
  }

  if (has_invalid_sources || has_invalid_targets || has_invalid_edges) {
    error("AnalysisGraph::remove_edges");

    if (has_invalid_sources) {
      cerr << "\tFollowing source vertexes are not in the CAG!" << endl;
      for (string invalid_src : invalid_sources) {
        cerr << "\t\t" << invalid_src << endl;
      }
    }

    if (has_invalid_targets) {
      cerr << "\tFollowing target vertexes are not in the CAG!" << endl;
      for (string invalid_tgt : invalid_targets) {
        cerr << "\t\t" << invalid_tgt << endl;
      }
    }

    if (has_invalid_edges) {
      cerr << "\tFollowing edges are not in the CAG!" << endl;
      for (pair<string, string> invalid_edge : invalid_edges) {
        cerr << "\t\t" << invalid_edge.first << " --to-> "
             << invalid_edge.second << endl;
      }
    }
  }
}


/*
 ============================================================================
 Public: Modify graph
 ============================================================================
*/

void AnalysisGraph::prune(int cutoff) {
  int num_verts = this->num_vertices();
  int src_degree = -1;
  int tgt_degree = -1;

  for (int tgt = 0; tgt < num_verts; ++tgt) {
    for (int src = 0; src < num_verts; ++src) {
      if (this->A_beta_factors[tgt][src] &&
          this->A_beta_factors[tgt][src]
              ->has_multiple_paths_longer_than_or_equal_to(cutoff)) {
        // src_degree = this->get_degree(src);
        // tgt_degree = this->get_degree(tgt);

        // if (src_degree != 1 && tgt_degree != 1) {
        // This check will always be true.
        // If there is a direct edge src --> tgt and
        // if there are multiple paths, then the degree
        // will always be > 1
        pair<int, int> edge = make_pair(src, tgt);

        // edge ≡ β
        if (in(this->beta2cell, edge)) {
          // There is a direct edge src --> tgt
          // Remove that edge
          boost::remove_edge(src, tgt, this->graph);
        }
      }
    }
  }
  // Recalculate all the directed simple paths
  this->find_all_paths();
}

void AnalysisGraph::merge_nodes(string concept_1,
                                string concept_2,
                                bool same_polarity) {
  // Check whetehr concept_1 and concept_2 are in the CAG
  this->get_vertex_id(concept_1);
  this->get_vertex_id(concept_2);

  for (int predecessor : this->predecessors(concept_1)) {
    Edge& edge_to_remove = this->edge(predecessor, concept_1);
    vector<Statement>& evidence_move = edge_to_remove.evidence;
    if (!same_polarity) {
      for (Statement stmt : edge_to_remove.evidence) {
        stmt.object.polarity = -stmt.object.polarity;
      }
    }

    // Add the edge predecessor --> vertex_to_keep
    auto edge_to_keep = this->add_edge(predecessor, concept_2).first;

    // Move all the evidence from vertex_delete to the
    // newly created (or existing) edge
    // predecessor --> vertex_to_keep
    vector<Statement>& evidence_keep = this->edge(edge_to_keep).evidence;

    evidence_keep.resize(evidence_keep.size() + evidence_move.size());

    move(evidence_move.begin(),
         evidence_move.end(),
         evidence_keep.end() - evidence_move.size());
  }

  for (int successor : this->successors(concept_1)) {

    // Get the edge descripter for
    //                   vertex_to_remove --> successor
    Edge& edge_to_remove = this->edge(concept_1, successor);
    vector<Statement>& evidence_move = edge_to_remove.evidence;

    if (!same_polarity) {
      for (Statement stmt : edge_to_remove.evidence) {
        stmt.subject.polarity = -stmt.subject.polarity;
      }
    }

    // Add the edge   successor --> vertex_to_keep
    auto edge_to_keep = this->add_edge(concept_2, successor).first;

    // Move all the evidence from vertex_delete to the
    // newly created (or existing) edge
    // vertex_to_keep --> successor
    vector<Statement>& evidence_keep = this->edge(edge_to_keep).evidence;

    evidence_keep.resize(evidence_keep.size() + evidence_move.size());

    move(evidence_move.begin(),
         evidence_move.end(),
         evidence_keep.end() - evidence_move.size());
  }

  // Remove vertex_to_remove from the CAG
  // Note: This is an overloaded private method that takes in a vertex id
  this->remove_node(concept_1);
}

void AnalysisGraph::change_polarity_of_edge(string source_concept,
                                            int source_polarity,
                                            string target_concept,
                                            int target_polarity) {
  int src_id = this->get_vertex_id(source_concept);
  int tgt_id = this->get_vertex_id(target_concept);

  pair<int, int> edge = make_pair(src_id, tgt_id);

  // edge ≡ β
  if (in(this->beta2cell, edge)) {
    // There is a edge from src_concept to tgt_concept
    // get that edge object
    auto e = boost::edge(src_id, tgt_id, this->graph).first;

    this->graph[e].change_polarity(source_polarity, target_polarity);
  }
}


/*
 ============================================================================
 Public: Populate graph with data
 ============================================================================
*/

void AnalysisGraph::set_indicator(string concept,
                                  string indicator,
                                  string source) {
  if (in(this->indicators_in_CAG, indicator)) {
    debug("{0} already exists in Causal Analysis Graph, Indicator {0} was "
          "not added to Concept {1}.",
          indicator,
          concept);
    return;
  }
  (*this)[concept].add_indicator(indicator, source);
  this->indicators_in_CAG.insert(indicator);
}

void AnalysisGraph::delete_indicator(string concept, string indicator) {
  (*this)[concept].delete_indicator(indicator);
  this->indicators_in_CAG.erase(indicator);
}

void AnalysisGraph::delete_all_indicators(string concept) {
  (*this)[concept].clear_indicators();
}

void AnalysisGraph::map_concepts_to_indicators(int n_indicators,
                                               string country) {
  spdlog::set_level(spdlog::level::debug);
  sqlite3* db = nullptr;
  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
  if (rc != SQLITE_OK) {
    throw runtime_error(
        "Could not open db. Do you have the DELPHI_DB "
        "environment correctly set to point to the Delphi database?");
  }
  sqlite3_stmt* stmt = nullptr;
  string query_base = "select Indicator from concept_to_indicator_mapping ";
  string query;

  // Check if there are any data values for an indicator for this country.
  auto has_data = [&](string indicator) {
    query =
        "select `Value` from indicator where `Variable` like '{0}' and `Country` like '{1}'"_format(
            indicator, country);
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt) == SQLITE_ROW;
    sqlite3_finalize(stmt);
    stmt = nullptr;
    return rc;
  };

  auto get_indicator_source = [&](string indicator) {
    query =
        "select `Source` from indicator where `Variable` like '{0}' and `Country` like '{1}' limit 1"_format(
            indicator, country);
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    string source =
        string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    sqlite3_finalize(stmt);
    stmt = nullptr;
    return source;
  };

  for (Node& node : this->nodes()) {
    node.clear_indicators(); // Clear pre-existing attached indicators

    query = "{0} where `Concept` like '{1}' order by `Score` desc"_format(
        query_base, node.name);
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);

    vector<string> matches = {};
    while (sqlite3_step(stmt) == SQLITE_ROW) {
      matches.push_back(
          string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))));
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;

    string ind_name, ind_source;

    for (int i = 0; i < n_indicators; i++) {
      bool at_least_one_indicator_found = false;
      for (string indicator : matches) {
        if (!in(this->indicators_in_CAG, indicator) and has_data(indicator)) {
          node.add_indicator(indicator, get_indicator_source(indicator));
          this->indicators_in_CAG.insert(indicator);
          at_least_one_indicator_found = true;
          break;
        }
      }
      if (!at_least_one_indicator_found) {
        debug("No suitable indicators found for concept '{0}' for country "
              "'{1}', please select "
              "one manually.",
              node.name,
              country);
      }
    }
  }
  rc = sqlite3_finalize(stmt);
  rc = sqlite3_close(db);
  stmt = nullptr;
  db = nullptr;
}

void AnalysisGraph::set_derivative(string concept, double derivative) {
  int v = this->get_vertex_id(concept);
  this->s0[2 * v + 1] = derivative;
}


/*
 ============================================================================
 Public: Utilities
 ============================================================================
*/

void AnalysisGraph::find_all_paths() {
  auto verts = this->node_indices();

  // Allocate the 2D array that keeps track of the cells of the transition
  // matrix (A_original) that are dependent on βs.
  // This function can be called anytime after creating the CAG.
  this->allocate_A_beta_factors();

  // Clear the multimap that keeps track of cells in the transition
  // matrix that are dependent on each β.
  this->beta2cell.clear();

  // Clear the set of all the β dependent cells
  this->beta_dependent_cells.clear();

  for_each(verts, [&](int start) {
    for_each(verts, [&](int end) {
      if (start != end) {
        this->find_all_paths_between(start, end);
      }
    });
  });

  // Allocate the cell value calculation data structures
  int num_verts = this->num_vertices();

  for (int row = 0; row < num_verts; ++row) {
    for (int col = 0; col < num_verts; ++col) {
      if (this->A_beta_factors[row][col]) {
        this->A_beta_factors[row][col]->allocate_datastructures();
      }
    }
  }
}

void AnalysisGraph::set_random_seed(int seed) {
  this->rng_instance = RNG::rng();
  this->rng_instance->set_seed(seed);
}


/*
 ============================================================================
 Public: Syntheitc data experiment
 ============================================================================
*/

pair<PredictedObservedStateSequence, Prediction>
AnalysisGraph::test_inference_with_synthetic_data(int start_year,
                                                  int start_month,
                                                  int end_year,
                                                  int end_month,
                                                  int res,
                                                  int burn,
                                                  string country,
                                                  string state,
                                                  string county,
                                                  map<string, string> units,
                                                  InitialBeta initial_beta) {
  synthetic_data_experiment = true;
  this->initialize_random_number_generator();

  this->n_timesteps = this->calculate_num_timesteps(
      start_year, start_month, end_year, end_month);
  this->init_betas_to(initial_beta);
  this->sample_initial_transition_matrix_from_prior();
  this->parameterize(country, state, county, start_year, start_month, units);

  // Initialize the latent state vector at time 0
  this->set_random_initial_latent_state();
  this->generate_synthetic_latent_state_sequence();
  this->generate_synthetic_observed_state_sequence_from_synthetic_latent_state_sequence();

  for (vector<vector<double>> obs : this->test_observed_state_sequence) {
    print("({}, {})\n", obs[0][0], obs[1][0]);
  }

  this->train_model(start_year,
                    start_month,
                    end_year,
                    end_month,
                    res,
                    burn,
                    country,
                    state,
                    county,
                    units,
                    InitialBeta::ZERO);

  return make_pair(
      this->test_observed_state_sequence,
      this->generate_prediction(start_year, start_month, end_year, end_month));

  RNG::release_instance();
  synthetic_data_experiment = false;
}


/*
 ============================================================================
 Public: Printing and Imaging
 ============================================================================
*/

void AnalysisGraph::print_nodes() {
  print("Vertex IDs and their names in the CAG\n");
  print("Vertex ID : Name\n");
  print("--------- : ----\n");
  for_each(this->node_indices(), [&](int v) {
    cout << v << "         " << this->graph[v].name << endl;
  });
}

void AnalysisGraph::print_edges() {
  for_each(edges(), [&](auto e) {
    cout << "(" << (*this)[boost::source(e, this->graph)].name << ", "
         << (*this)[boost::target(e, this->graph)].name << ")" << endl;
  });
}

void AnalysisGraph::print_name_to_vertex() {
  for (auto [name, vert] : this->name_to_vertex) {
    cout << name << " -> " << vert << endl;
  }
  cout << endl;
}

void AnalysisGraph::print_indicators() {
  for (int v : this->node_indices()) {
    cout << v << ":" << (*this)[v].name << endl;
    for (auto [name, vert] : (*this)[v].nameToIndexMap) {
      cout << "\t"
           << "indicator " << vert << ": " << name << endl;
    }
  }
}

void AnalysisGraph::print_A_beta_factors() {
  int num_verts = this->num_vertices();

  for (int row = 0; row < num_verts; ++row) {
    for (int col = 0; col < num_verts; ++col) {
      cout << endl << "Printing cell: (" << row << ", " << col << ") " << endl;
      if (this->A_beta_factors[row][col]) {
        this->A_beta_factors[row][col]->print_beta2product();
      }
    }
  }
}

void AnalysisGraph::print_all_paths() {
  int num_verts = this->num_vertices();

  if (this->A_beta_factors.size() != num_verts ||
      this->A_beta_factors[0].size() != num_verts) {
    this->find_all_paths();
  }

  cout << "All the simple paths of:" << endl;

  for (int row = 0; row < num_verts; ++row) {
    for (int col = 0; col < num_verts; ++col) {
      if (this->A_beta_factors[row][col]) {
        this->A_beta_factors[row][col]->print_paths();
      }
    }
  }
}

// Given an edge (source, target vertex ids - i.e. a β ≡ ∂target/∂source),
// print all the transition matrix cells that are dependent on it.
void AnalysisGraph::print_cells_affected_by_beta(int source, int target) {
  typedef multimap<pair<int, int>, pair<int, int>>::iterator MMapIterator;

  pair<int, int> beta = make_pair(source, target);

  pair<MMapIterator, MMapIterator> beta_dept_cells =
      this->beta2cell.equal_range(beta);

  cout << endl
       << "Cells of A afected by beta_(" << source << ", " << target << ")"
       << endl;

  for (MMapIterator it = beta_dept_cells.first; it != beta_dept_cells.second;
       it++) {
    cout << "(" << it->second.first * 2 << ", " << it->second.second * 2 + 1
         << ") ";
  }
  cout << endl;
}
