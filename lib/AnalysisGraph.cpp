#include "AnalysisGraph.hpp"
#include "data.hpp"
#include "Node.hpp"
#include "tqdm.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/filtered.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/algorithm/sort.hpp>
#include <cmath>
#include <range/v3/all.hpp>
#include <sqlite3.h>
#include <type_traits>
#include "spdlog/spdlog.h"
#include "dbg.h"

using namespace std;
using boost::for_each;
using boost::make_iterator_range;
using fmt::print, fmt::format;
using tq::tqdm;
using namespace fmt::literals;
using ranges::views::filter;
using spdlog::debug;
using spdlog::error;
using spdlog::warn;
using namespace delphi::utils;

typedef multimap<pair<int, int>, pair<int, int>>::iterator MMAPIterator;

size_t AnalysisGraph::num_vertices() {
  return boost::num_vertices(this->graph);
}

size_t AnalysisGraph::num_edges() { return boost::num_edges(this->graph); }

Node& AnalysisGraph::operator[](int index) { return this->graph[index]; }

Node& AnalysisGraph::operator[](string node_name) {
  return (*this)[this->name_to_vertex.at(node_name)];
}

auto AnalysisGraph::successors(int i) {
  return make_iterator_range(boost::adjacent_vertices(i, this->graph));
}

auto AnalysisGraph::successors(string node_name) {
  return this->successors(this->name_to_vertex.at(node_name));
}

auto AnalysisGraph::predecessors(int i) {
  return make_iterator_range(boost::inv_adjacent_vertices(i, this->graph));
}

auto AnalysisGraph::predecessors(string node_name) {
  return this->predecessors(this->name_to_vertex.at(node_name));
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

void AnalysisGraph::map_concepts_to_indicators(int n_indicators,
                                               string country) {
  using boost::range::sort;
  spdlog::set_level(spdlog::level::debug);
  sqlite3* db;
  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
  if (rc) {
    throw("Could not open db. Do you have the DELPHI_DB "
          "environment correctly set to point to the Delphi database?");
  }
  sqlite3_stmt* stmt;
  string query_base = "select Indicator from concept_to_indicator_mapping ";
  string query;

  // Check if there are any data values for an indicator for this country.
  auto has_data = [&](string indicator) {
    query =
        "select `Value` from indicator where `Variable` like '{0}' and `Country` like '{1}'"_format(
            indicator, country);
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt) == SQLITE_ROW;
    sqlite3_reset(stmt);
    return rc;
  };

  auto get_indicator_source = [&](string indicator) {
    query =
        "select `Source` from indicator where `Variable` like '{0}' and `Country` like '{1}' limit 1"_format(
            indicator, country);
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    string source = string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    sqlite3_reset(stmt);
    return source;
  };

  // Making the mapping more deterministic by sorting the nodes alphabetically
  // by name.

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
    sqlite3_reset(stmt);

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


void AnalysisGraph::allocate_A_beta_factors() {
  this->A_beta_factors.clear();

  int num_verts = this->num_vertices();

  for (int vert = 0; vert < num_verts; ++vert) {
    this->A_beta_factors.push_back(
        vector<shared_ptr<Tran_Mat_Cell>>(num_verts));
  }
}

void AnalysisGraph::get_subgraph(int vert,
                                 unordered_set<int>& vertices_to_keep,
                                 int cutoff,
                                 bool inward) {

  // Mark the current vertex visited
  (*this)[vert].visited = true;
  vertices_to_keep.insert(vert);

  if (cutoff != 0) {
    cutoff--;

    // Recursively process all the vertices adjacent to the current vertex
    if (inward) {
      for_each(this->predecessors(vert), [&](int v) {
        if (!(*this)[v].visited) {
          this->get_subgraph(v, vertices_to_keep, cutoff, inward);
        }
      });
    }
    else {
      for_each(this->successors(vert), [&](int v) {
        if (!(*this)[v].visited) {
          this->get_subgraph(v, vertices_to_keep, cutoff, inward);
        }
      });
    }
  }

  // Mark the current vertex unvisited
  (*this)[vert].visited = false;
};

void AnalysisGraph::get_subgraph_between(int start,
                                         int end,
                                         vector<int>& path,
                                         unordered_set<int>& vertices_to_keep,
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
    vertices_to_keep.insert(path.begin(), path.end());
  }
  else if (cutoff != 0) {
    cutoff--;

    // Recursively process all the vertices adjacent to the current vertex
    for_each(this->successors(start), [&](int v) {
      if (!(*this)[v].visited) {
        this->get_subgraph_between(v, end, path, vertices_to_keep, cutoff);
      }
    });
  }

  // Remove current vertex from the path and make it unvisited
  path.pop_back();
  (*this)[start].visited = false;
};

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
};

void AnalysisGraph::set_default_initial_state() {
  // Let vertices of the CAG be v = 0, 1, 2, 3, ...
  // Then,
  //    indexes 2*v keeps track of the state of each variable v
  //    indexes 2*v+1 keeps track of the state of ∂v/∂t
  int num_verts = this->num_vertices();
  int num_els = num_verts * 2;

  this->s0 = Eigen::VectorXd(num_els);
  this->s0.setZero();

  for (int i = 0; i < num_els; i += 2) {
    this->s0(i) = 1.0;
  }
}

int AnalysisGraph::get_vertex_id_for_concept(string concept, string caller) {
  int vert_id = -1;

  try {
    vert_id = this->name_to_vertex.at(concept);
  }
  catch (const out_of_range& oor) {
    error("AnalysisGraph::{0}\n"
          "The concept {1} is not in the CAG!",
          caller,
          concept);
    rethrow_exception(current_exception());
  }

  return vert_id;
}

int AnalysisGraph::get_degree(int vertex_id) {
  return boost::in_degree(vertex_id, this->graph) +
         boost::out_degree(vertex_id, this->graph);
}

AnalysisGraph AnalysisGraph::from_json_file(string filename,
                                            double belief_score_cutoff,
                                            double grounding_score_cutoff,
                                            string ontology) {
  auto json_data = delphi::utils::load_json(filename);
  debug("Loading INDRA statements JSON file.");

  AnalysisGraph G;

  unordered_map<string, int> name_to_vertex = {};

  debug("Processing INDRA statements...");
  for (auto stmt : tqdm(json_data)) {
    if (stmt["type"] == "Influence") {
      auto subj_ground = stmt["subj"]["concept"]["db_refs"][ontology][0][1];
      auto obj_ground = stmt["obj"]["concept"]["db_refs"][ontology][0][1];
      bool grounding_check = (subj_ground >= grounding_score_cutoff) and
                             (obj_ground >= grounding_score_cutoff);
      if (grounding_check) {
        auto subj = stmt["subj"]["concept"]["db_refs"]["WM"][0][0];
        auto obj = stmt["obj"]["concept"]["db_refs"]["WM"][0][0];
        if (!subj.is_null() and !obj.is_null()) {
          if (stmt["belief"] < belief_score_cutoff) {
            continue;
          }

          string subj_str = subj.get<string>();
          string obj_str = obj.get<string>();

          if (subj_str.compare(obj_str) != 0) { // Guard against self loops
            // Add the nodes to the graph if they are not in it already
            for (string name : {subj_str, obj_str}) {
              G.add_node(name);
            }

            // Add the edge to the graph if it is not in it already
            for (auto evidence : stmt["evidence"]) {
              auto annotations = evidence["annotations"];
              auto subj_adjectives = annotations["subj_adjectives"];
              auto obj_adjectives = annotations["obj_adjectives"];
              auto subj_adjective =
                  (!subj_adjectives.is_null() and subj_adjectives.size() > 0)
                      ? subj_adjectives[0]
                      : "None";
              auto obj_adjective =
                  (obj_adjectives.size() > 0) ? obj_adjectives[0] : "None";
              auto subj_polarity = annotations["subj_polarity"];
              auto obj_polarity = annotations["obj_polarity"];

              if (subj_polarity.is_null()) {
                subj_polarity = 1;
              }
              if (obj_polarity.is_null()) {
                obj_polarity = 1;
              }
              string subj_adj_str = subj_adjective.get<string>();
              string obj_adj_str = subj_adjective.get<string>();
              auto causal_fragment =
                  CausalFragment({subj_adj_str, subj_polarity, subj_str},
                                 {obj_adj_str, obj_polarity, obj_str});

              G.add_edge(causal_fragment);
            }
          }
        }
      }
    }
  }
  G.initialize_random_number_generator();
  return G;
}

AnalysisGraph
AnalysisGraph::from_uncharted_json_dict(nlohmann::json json_data) {
  AnalysisGraph G;

  unordered_map<string, int> name_to_vertex = {};

  auto statements = json_data["statements"];

  for (auto stmt : statements) {
    auto evidence = stmt["evidence"];

    if (evidence.is_null()) {
      continue;
    }

    auto subj = stmt["subj"];
    auto obj = stmt["obj"];

    if (subj.is_null() or obj.is_null()) {
      continue;
    }

    auto subj_db_ref = subj["db_refs"];
    auto obj_db_ref = obj["db_refs"];

    if (subj_db_ref.is_null() or obj_db_ref.is_null()) {
      continue;
    }

    auto subj_concept_json = subj_db_ref["concept"];
    auto obj_concept_json = obj_db_ref["concept"];

    if (subj_concept_json.is_null() or obj_concept_json.is_null()) {
      continue;
    }

    // TODO: Not sure why python version is doing a split on / and
    // then again a join on /!!
    string subj_name = subj_concept_json.get<string>();
    string obj_name = obj_concept_json.get<string>();

    auto subj_delta = stmt["subj_delta"];
    auto obj_delta = stmt["obj_delta"];

    auto subj_polarity_json = subj_delta["polarity"];
    auto obj_polarity_json = obj_delta["polarity"];

    int subj_polarity = 1;
    int obj_polarity = 1;
    if (!subj_polarity_json.is_null()) {
      subj_polarity = subj_polarity_json.get<int>();
    }

    if (!obj_polarity_json.is_null()) {
      obj_polarity = obj_polarity_json.get<int>();
    }

    auto subj_adjectives = subj_delta["adjectives"];
    auto obj_adjectives = obj_delta["adjectives"];
    auto subj_adjective =
        (!subj_adjectives.is_null() and subj_adjectives.size() > 0)
            ? subj_adjectives[0]
            : "None";
    auto obj_adjective =
        (obj_adjectives.size() > 0) ? obj_adjectives[0] : "None";

    string subj_adj_str = subj_adjective.get<string>();
    string obj_adj_str = obj_adjective.get<string>();

    auto causal_fragment =
        CausalFragment({subj_adj_str, subj_polarity, subj_name},
                       {obj_adj_str, obj_polarity, obj_name});
    string text = stmt["evidence"][0]["text"].get<string>();
    G.add_edge(causal_fragment);
  }
  G.initialize_random_number_generator();
  return G;
}

AnalysisGraph AnalysisGraph::from_uncharted_json_string(string json_string) {
  auto json_data = nlohmann::json::parse(json_string);
  return AnalysisGraph::from_uncharted_json_dict(json_data);
}

AnalysisGraph AnalysisGraph::from_uncharted_json_file(string filename) {
  auto json_data = load_json(filename);
  return AnalysisGraph::from_uncharted_json_dict(json_data);
}

void AnalysisGraph::clear_state() {
  this->A_beta_factors.clear();

  // Clear the multimap that keeps track of cells in the transition
  // matrix that are dependent on each β.
  this->beta2cell.clear();

  // Clear the set of all the β dependent cells
  this->beta_dependent_cells.clear();
}

AnalysisGraph AnalysisGraph::get_subgraph_for_concept(string concept,
                                                      bool inward,
                                                      int depth) {

  using ranges::views::filter, ranges::views::transform, ranges::to;

  // Mark all the vertices as not visited
  for_each(this->nodes(), [](Node& node) { node.visited = false; });

  int num_verts = this->num_vertices();

  unordered_set<int> vertices_to_keep = unordered_set<int>();

  this->get_subgraph(
      this->name_to_vertex.at(concept), vertices_to_keep, depth, inward);

  unordered_set<string> nodes_to_remove =
      this->node_indices() |
      filter([&](int v) { return !in(vertices_to_keep, v); }) |
      transform([&](int v) { return (*this)[v].name; }) | to<unordered_set>();

  if (vertices_to_keep.size() == 0) {
    warn("Subgraph has 0 nodes - returning an empty CAG!");
  }

  // Make a copy of current AnalysisGraph
  // TODO: We have to make sure that we are making a deep copy.
  //       Test so far does not show suspicious behavior
  AnalysisGraph G_sub = *this;
  for_each(nodes_to_remove, [&](string n) { G_sub.remove_node(n); });
  G_sub.clear_state();

  return G_sub;
}

AnalysisGraph AnalysisGraph::get_subgraph_for_concept_pair(
    string source_concept, string target_concept, int cutoff) {
  int src_id = this->get_vertex_id_for_concept(
      source_concept, "get_subgraph_for_concept_pair()");
  int tgt_id = this->get_vertex_id_for_concept(
      target_concept, "get_subgraph_for_concept_pair()");

  unordered_set<int> vertices_to_keep;
  unordered_set<string> vertices_to_remove;
  vector<int> path;

  // Mark all the vertices are not visited
  for_each(this->node_indices(), [&](int v) { (*this)[v].visited = false; });

  this->get_subgraph_between(src_id, tgt_id, path, vertices_to_keep, cutoff);

  if (vertices_to_keep.size() == 0) {
    warn("AnalysisGraph::get_subgraph_for_concept_pair(): "
         "There are no paths of length <= {0} from "
         "source concept {1} --to-> target concept {2}. "
         "Returning an empty CAG!",
         cutoff,
         source_concept,
         target_concept);
  }

  // Determine the vertices to be removed
  for (int vert_id : this->node_indices()) {
    if (in(vertices_to_keep, vert_id)) {
      vertices_to_remove.insert((*this)[vert_id].name);
    }
  }

  // Make a copy of current AnalysisGraph
  // TODO: We have to make sure that we are making a deep copy.
  //       Test so far does not show suspicious behavior
  AnalysisGraph G_sub = *this;
  G_sub.remove_nodes(vertices_to_remove);
  G_sub.clear_state();

  return G_sub;
}

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
        //}
      }
    }
  }
  // Recalculate all the directed simple paths
  this->find_all_paths();
}

void AnalysisGraph::remove_node(int node_id) {
  // Delete all the edges incident to this node
  boost::clear_vertex(node_id, this->graph);

  // Remove the vertex
  boost::remove_vertex(node_id, this->graph);

  // Update the internal meta-data
  for (int vert_id : this->node_indices()) {
    this->name_to_vertex.at((*this)[vert_id].name) = vert_id;
  }
}

void AnalysisGraph::remove_node(string concept) {
  this->remove_node(this->name_to_vertex.at(concept));
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
    error("AnalysisGraph::remove_vertex()\n"
          "\tThe following concepts were not present in the CAG!\n");
    for (string invalid_concept : invalid_concepts) {
      cerr << "\t\t" << invalid_concept << endl;
    }
  }
}

void AnalysisGraph::remove_edge(string src, string tgt) {
  int src_id = -1;
  int tgt_id = -1;

  try {
    src_id = this->name_to_vertex.at(src);
  }
  catch (const out_of_range& oor) {
    error("AnalysisGraph::remove_edge: Source vertex {} is not in the CAG!",
          src);
    return;
  }

  try {
    tgt_id = this->name_to_vertex.at(tgt);
  }
  catch (const out_of_range& oor) {
    error("AnalysisGraph::remove_edge: \n"
          "\tTarget vertex {} is not in the CAG!",
          tgt);
    return;
  }

  // Remove the edge
  boost::remove_edge(src_id, tgt_id, this->graph);
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
                     src_id = this->name_to_vertex.at(edge.first);
                   }
                   catch (const out_of_range& oor) {
                     src_id = -1;
                   }

                   // Flag invalid target vertices
                   try {
                     tgt_id = this->name_to_vertex.at(edge.second);
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

/** Output the graph in DOT format */
string AnalysisGraph::to_dot() {
  auto [G, gvc] = this->to_agraph();

  stringstream sstream;
  stringbuf* sstream_buffer;
  streambuf* original_cout_buffer;

  // Back up original cout buffer
  original_cout_buffer = cout.rdbuf();
  sstream_buffer = sstream.rdbuf();

  // Redirect cout to sstream
  cout.rdbuf(sstream_buffer);

  gvRender(gvc, G, "dot", stdout);
  agclose(G);
  gvFreeContext(gvc);

  // Restore cout's original buffer
  cout.rdbuf(original_cout_buffer);

  // Return the string with the graph in DOT format
  return sstream.str();
}

AnalysisGraph
AnalysisGraph::from_causal_fragments(vector<CausalFragment> causal_fragments) {
  AnalysisGraph G;

  for (CausalFragment cf : causal_fragments) {
    Event subject = Event(cf.first);
    Event object = Event(cf.second);

    string subj_name = subject.concept_name;
    string obj_name = object.concept_name;

    if (subj_name.compare(obj_name) != 0) { // Guard against self loops
      // Add the nodes to the graph if they are not in it already
      for (string name : {subj_name, obj_name}) {
        G.add_node(name);
      }
      G.add_edge(cf);
    }
  }
  G.initialize_random_number_generator();
  return G;
}

Edge& AnalysisGraph::edge(int source, int target) {
  return this->graph[boost::edge(source, target, this->graph).first];
}

Edge& AnalysisGraph::edge(int source, string target) {
  return this->graph
      [boost::edge(source, this->name_to_vertex.at(target), this->graph).first];
}

Edge& AnalysisGraph::edge(string source, int target) {
  return this->graph
      [boost::edge(this->name_to_vertex.at(source), target, this->graph).first];
}

Edge& AnalysisGraph::edge(string source, string target) {
  return this->graph[boost::edge(this->name_to_vertex.at(source),
                                 this->name_to_vertex.at(target),
                                 this->graph)
                         .first];
}

Edge& AnalysisGraph::edge(boost::graph_traits<DiGraph>::edge_descriptor e) {
  return this->graph[e];
}

pair<boost::graph_traits<DiGraph>::edge_descriptor, bool>
AnalysisGraph::add_edge(int source, int target) {
  return boost::add_edge(source, target, this->graph);
}

pair<boost::graph_traits<DiGraph>::edge_descriptor, bool>
AnalysisGraph::add_edge(int source, string target) {
  return boost::add_edge(source, this->name_to_vertex.at(target), this->graph);
}

pair<boost::graph_traits<DiGraph>::edge_descriptor, bool>
AnalysisGraph::add_edge(string source, int target) {
  return boost::add_edge(this->name_to_vertex.at(source), target, this->graph);
}

pair<boost::graph_traits<DiGraph>::edge_descriptor, bool>
AnalysisGraph::add_edge(string source, string target) {
  return boost::add_edge(this->name_to_vertex.at(source),
                         this->name_to_vertex.at(target),
                         this->graph);
}

void AnalysisGraph::merge_nodes(string concept_1,
                                string concept_2,
                                bool same_polarity) {
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

void AnalysisGraph::set_log_likelihood() {
  this->previous_log_likelihood = this->log_likelihood;
  this->log_likelihood = 0.0;

  for (int ts = 0; ts < this->n_timesteps; ts++) {
    this->set_current_latent_state(ts);

    // Access
    // observed_state[ vertex ][ indicator ]
    const vector<vector<vector<double>>>& observed_state =
        this->observed_state_sequence[ts];

    for (int v : this->node_indices()) {
      const int& num_inds_for_v = observed_state[v].size();

      for (int i = 0; i < observed_state[v].size(); i++) {
        const Indicator& ind = this->graph[v].indicators[i];
        for (int j = 0; j < observed_state[v][i].size(); j++) {
          const double& value = observed_state[v][i][j];
          // Even indices of latent_state keeps track of the state of each
          // vertex
          double log_likelihood_component = this->log_normpdf(
              value, this->current_latent_state[2 * v] * ind.mean, ind.stdev);
          this->log_likelihood += log_likelihood_component;
        }
      }
    }
  }
}

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

void AnalysisGraph::print_indicators() {
  for (int v : this->node_indices()) {
    cout << v << ":" << (*this)[v].name << endl;
    for (auto [name, vert] : (*this)[v].nameToIndexMap) {
      cout << "\t"
           << "indicator " << vert << ": " << name << endl;
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

void AnalysisGraph::print_name_to_vertex() {
  for (auto [name, vert] : this->name_to_vertex) {
    cout << name << " -> " << vert << endl;
  }
  cout << endl;
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

void AnalysisGraph::change_polarity_of_edge(string source_concept,
                                            int source_polarity,
                                            string target_concept,
                                            int target_polarity) {
  int src_id = this->get_vertex_id_for_concept(source_concept,
                                               "change_polarity_of_edge");
  int tgt_id = this->get_vertex_id_for_concept(target_concept,
                                               "change_polarity_of_edge");

  pair<int, int> edge = make_pair(src_id, tgt_id);

  // edge ≡ β
  if (in(this->beta2cell, edge)) {
    // There is a edge from src_concept to tgt_concept
    // get that edge object
    auto e = boost::edge(src_id, tgt_id, this->graph).first;

    this->graph[e].change_polarity(source_polarity, target_polarity);
  }
}

// Given an edge (source, target vertex ids - i.e. a β ≡ ∂target/∂source),
// print all the transition matrix cells that are dependent on it.
void AnalysisGraph::print_cells_affected_by_beta(int source, int target) {
  typedef multimap<pair<int, int>, pair<int, int>>::iterator MMAPIterator;

  pair<int, int> beta = make_pair(source, target);

  pair<MMAPIterator, MMAPIterator> beta_dept_cells =
      this->beta2cell.equal_range(beta);

  cout << endl
       << "Cells of A afected by beta_(" << source << ", " << target << ")"
       << endl;

  for (MMAPIterator it = beta_dept_cells.first; it != beta_dept_cells.second;
       it++) {
    cout << "(" << it->second.first * 2 << ", " << it->second.second * 2 + 1
         << ") ";
  }
  cout << endl;
}

// Sample elements of the stochastic transition matrix from the
// prior distribution, based on gradable adjectives.
void AnalysisGraph::sample_initial_transition_matrix_from_prior() {
  int num_verts = this->num_vertices();

  // A base transition matrix with the entries that does not change across
  // samples.
  /*
   *          0  1  2  3  4  5
   *  var_1 | 1 Δt             | 0
   *        | 0  1  0  0  0  0 | 1 ∂var_1 / ∂t
   *  var_2 |       1 Δt       | 2
   *        | 0  0  0  1  0  0 | 3
   *  var_3 |             1 Δt | 4
   *        | 0  0  0  0  0  1 | 5
   *
   *  Based on the directed simple paths in the CAG, some of the remaining
   *  blank cells would be filled with β related values
   *  If we include second order derivatives to the model, there would be
   *  three rows for each variable and some of the off diagonal elements
   *  of rows with index % 3 = 1 would be non zero.
   */
  this->A_original = Eigen::MatrixXd::Identity(num_verts * 2, num_verts * 2);

  // Fill the Δts
  for (int vert = 0; vert < 2 * num_verts; vert += 2) {
    this->A_original(vert, vert + 1) = this->delta_t;
  }

  // Update the β factor dependent cells of this matrix
  for (auto& [row, col] : this->beta_dependent_cells) {
    this->A_original(row * 2, col * 2 + 1) =
        this->A_beta_factors[row][col]->compute_cell(this->graph);
  }
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

void AnalysisGraph::set_initial_latent_state_from_observed_state_sequence() {
  int num_verts = this->num_vertices();

  this->set_default_initial_state();

  for (int v = 0; v < num_verts; v++) {
    vector<Indicator>& indicators = (*this)[v].indicators;
    vector<double> next_state_values;
    for (int i = 0; i < indicators.size(); i++) {
      Indicator& ind = indicators[i];

      double ind_mean = ind.get_mean();

      while (ind_mean == 0) {
        ind_mean = this->norm_dist(this->rand_num_generator);
      }
      double next_ind_value;
      if (this->observed_state_sequence[1][v][i].empty()) {
        next_ind_value = 0;
      }
      else {
        next_ind_value =
            delphi::utils::mean(this->observed_state_sequence[1][v][i]);
      }
      next_state_values.push_back(next_ind_value / ind_mean);
    }
    double diff = delphi::utils::mean(next_state_values) - this->s0(2 * v);
    this->s0(2 * v + 1) = diff;
  }
}

void AnalysisGraph::set_random_initial_latent_state() {
  int num_verts = this->num_vertices();

  this->set_default_initial_state();

  for (int v = 0; v < num_verts; v++) {
    this->s0(2 * v + 1) = 0.1 * this->uni_dist(this->rand_num_generator);
  }
}

void AnalysisGraph::init_betas_to(InitialBeta ib) {
  using boost::graph_traits;
  switch (ib) {
  // Initialize the initial β for this edge
  // Note: I am repeating the loop within each case for efficiency.
  // If we embed the switch withn the for loop, there will be less code
  // but we will evaluate the switch for each iteration through the loop
  case InitialBeta::ZERO:
    for (graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
      graph[e].beta = 0;
    }
    break;
  case InitialBeta::ONE:
    for (graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
      graph[e].beta = 1.0;
    }
    break;
  case InitialBeta::HALF:
    for (graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
      graph[e].beta = 0.5;
    }
    break;
  case InitialBeta::MEAN:
    for (graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
      graph[e].beta = graph[e].kde.mu;
    }
    break;
  case InitialBeta::RANDOM:
    for (graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
      // this->uni_dist() gives a random number in range [0, 1]
      // Multiplying by 2 scales the range to [0, 2]
      // Sustracting 1 moves the range to [-1, 1]
      graph[e].beta = this->uni_dist(this->rand_num_generator) * 2 - 1;
    }
    break;
  }
}

void AnalysisGraph::sample_predicted_latent_state_sequences(
    int prediction_timesteps,
    int initial_prediction_step,
    int total_timesteps) {
  this->n_timesteps = prediction_timesteps;

  int num_verts = this->num_vertices();

  // Allocate memory for prediction_latent_state_sequences
  this->predicted_latent_state_sequences.clear();
  this->predicted_latent_state_sequences = vector<vector<Eigen::VectorXd>>(
      this->res,
      vector<Eigen::VectorXd>(this->n_timesteps,
                              Eigen::VectorXd(num_verts * 2)));

  for (int samp = 0; samp < this->res; samp++) {
    int pred_step = initial_prediction_step;
    for (int ts = 0; ts < this->n_timesteps; ts++) {
      const Eigen::MatrixXd& A_t =
          0.1 * pred_step * this->transition_matrix_collection[samp];
      this->predicted_latent_state_sequences[samp][ts] = A_t.exp() * this->s0;
      pred_step++;
    }
  }
}

void AnalysisGraph::
    generate_predicted_observed_state_sequences_from_predicted_latent_state_sequences() {
  using ranges::to;
  using ranges::views::transform;

  // Allocate memory for observed_state_sequences
  this->predicted_observed_state_sequences.clear();
  this->predicted_observed_state_sequences =
      vector<PredictedObservedStateSequence>(
          this->res,
          PredictedObservedStateSequence(this->n_timesteps,
                                         vector<vector<double>>()));

  for (int samp = 0; samp < this->res; samp++) {
    vector<Eigen::VectorXd>& sample =
        this->predicted_latent_state_sequences[samp];

    this->predicted_observed_state_sequences[samp] =
        sample | transform([this](Eigen::VectorXd latent_state) {
          return this->sample_observed_state(latent_state);
        }) |
        to<vector>();
  }
}

Prediction AnalysisGraph::generate_prediction(int start_year,
                                              int start_month,
                                              int end_year,
                                              int end_month) {
  if (!this->trained) {
    print("Passed untrained Causal Analysis Graph (CAG) Model. \n",
          "Try calling <CAG>.train_model(...) first!");
    throw "Model not yet trained";
  }

  if (start_year < this->training_range.first.first ||
      (start_year == this->training_range.first.first &&
       start_month < this->training_range.first.second)) {
    print("The initial prediction date can't be before the\n"
          "inital training date. Defaulting initial prediction date\n"
          "to initial training date.");
    start_year = this->training_range.first.first;
    start_month = this->training_range.first.first;
  }

  /*
   *              total_timesteps
   *   ____________________________________________
   *  |                                            |
   *  v                                            v
   * start trainig                                 end prediction
   *  |--------------------------------------------|
   *  :           |--------------------------------|
   *  :         start prediction                   :
   *  ^           ^                                ^
   *  |___________|________________________________|
   *      diff              pred_timesteps
   */
  int total_timesteps =
      this->calculate_num_timesteps(this->training_range.first.first,
                                    this->training_range.first.second,
                                    end_year,
                                    end_month);

  this->pred_timesteps = this->calculate_num_timesteps(
      start_year, start_month, end_year, end_month);

  int pred_init_timestep = total_timesteps - pred_timesteps;

  int year = start_year;
  int month = start_month;

  this->pred_range.clear();
  this->pred_range = vector<string>(this->pred_timesteps);

  for (int ts = 0; ts < this->pred_timesteps; ts++) {
    this->pred_range[ts] = to_string(year) + "-" + to_string(month);

    if (month == 12) {
      year++;
      month = 1;
    }
    else {
      month++;
    }
  }

  this->sample_predicted_latent_state_sequences(
      this->pred_timesteps, pred_init_timestep, total_timesteps);
  this->generate_predicted_observed_state_sequences_from_predicted_latent_state_sequences();

  return make_tuple(
      this->training_range, this->pred_range, this->format_prediction_result());
}

FormattedPredictionResult AnalysisGraph::format_prediction_result() {
  // Access
  // [ sample ][ time_step ][ vertex_name ][ indicator_name ]
  auto result = FormattedPredictionResult(
      this->res,
      vector<unordered_map<string, unordered_map<string, double>>>(
          this->pred_timesteps));

  for (int samp = 0; samp < this->res; samp++) {
    for (int ts = 0; ts < this->pred_timesteps; ts++) {
      for (auto [vert_name, vert_id] : this->name_to_vertex) {
        for (auto [ind_name, ind_id] : (*this)[vert_id].nameToIndexMap) {
          result[samp][ts][vert_name][ind_name] =
              this->predicted_observed_state_sequences[samp][ts][vert_id]
                                                      [ind_id];
        }
      }
    }
  }

  return result;
}

vector<vector<double>> AnalysisGraph::prediction_to_array(string indicator) {
  int vert_id = -1;
  int ind_id = -1;

  auto result =
      vector<vector<double>>(this->res, vector<double>(this->pred_timesteps));

  // Find the vertex id the indicator is attached to and
  // the indicator id of it.
  // TODO: We can make this more efficient by making indicators_in_CAG
  // a map from indicator names to vertices they are attached to.
  // This is just a quick and dirty implementation
  for (auto [v_name, v_id] : this->name_to_vertex) {
    for (auto [i_name, i_id] : (*this)[v_id].nameToIndexMap) {
      if (indicator.compare(i_name) == 0) {
        vert_id = v_id;
        ind_id = i_id;
        goto indicator_found;
      }
    }
  }
  // Program will reach here only if the indicator is not found
  throw IndicatorNotFoundException(format(
      "AnalysisGraph::prediction_to_array - indicator \"{}\" not found!\n",
      indicator));

indicator_found:

  for (int samp = 0; samp < this->res; samp++) {
    for (int ts = 0; ts < this->pred_timesteps; ts++) {
      result[samp][ts] =
          this->predicted_observed_state_sequences[samp][ts][vert_id][ind_id];
    }
  }

  return result;
}

void AnalysisGraph::generate_synthetic_latent_state_sequence() {
  int num_verts = this->num_vertices();

  // Allocate memory for synthetic_latent_state_sequence
  this->synthetic_latent_state_sequence.clear();
  this->synthetic_latent_state_sequence = vector<Eigen::VectorXd>(
      this->n_timesteps, Eigen::VectorXd(num_verts * 2));

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
      transform([this](Eigen::VectorXd latent_state) {
        return this->sample_observed_state(latent_state);
      }) |
      to<vector>();
}

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

  synthetic_data_experiment = false;
}

vector<vector<double>>
AnalysisGraph::sample_observed_state(Eigen::VectorXd latent_state) {
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
    std::transform(indicators.begin(),
                   indicators.end(),
                   observed_state[v].begin(),
                   [&](Indicator ind) {
                     normal_distribution<double> gaussian(
                         ind.mean * latent_state[2 * v], ind.stdev);

                     //dbg(ind.get_name());
                     //dbg(ind.mean);
                     //dbg(ind.stdev);
                     return gaussian(this->rand_num_generator);
                   });
  }

  return observed_state;
}

void AnalysisGraph::update_transition_matrix_cells(
    boost::graph_traits<DiGraph>::edge_descriptor e) {
  pair<int, int> beta =
      make_pair(boost::source(e, this->graph), boost::target(e, this->graph));

  pair<MMAPIterator, MMAPIterator> beta_dept_cells =
      this->beta2cell.equal_range(beta);

  // TODO: I am introducing this to implement calculate_Δ_log_prior
  // Remember the cells of A that got changed and their previous values
  // this->A_cells_changed.clear();

  for (MMAPIterator it = beta_dept_cells.first; it != beta_dept_cells.second;
       it++) {
    int& row = it->second.first;
    int& col = it->second.second;

    // Note that I am remembering row and col instead of 2*row and 2*col+1
    // row and col resembles an edge in the CAG: row -> col
    // ( 2*row, 2*col+1 ) is the transition mateix cell that got changed.
    // this->A_cells_changed.push_back( make_tuple( row, col, A( row * 2, col
    // * 2 + 1 )));

    this->A_original(row * 2, col * 2 + 1) =
        this->A_beta_factors[row][col]->compute_cell(this->graph);
  }
}

void AnalysisGraph::sample_from_proposal() {
  // Randomly pick an edge ≡ β
  boost::iterator_range edge_it = this->edges();

  vector<boost::graph_traits<DiGraph>::edge_descriptor> e(1);
  sample(
      edge_it.begin(), edge_it.end(), e.begin(), 1, this->rand_num_generator);

  // Remember the previous β
  this->previous_beta = make_pair(e[0], this->graph[e[0]].beta);

  // Perturb the β
  // TODO: Check whether this perturbation is accurate
  graph[e[0]].beta += this->norm_dist(this->rand_num_generator);

  this->update_transition_matrix_cells(e[0]);
}

void AnalysisGraph::set_current_latent_state(int ts) {
  const Eigen::MatrixXd& A_t = ts * this->A_original;
  this->current_latent_state = A_t.exp() * this->s0;
}

double AnalysisGraph::log_normpdf(double x, double mean, double sd) {
  double var = pow(sd, 2);
  double log_denom = -0.5 * log(2 * M_PI) - log(sd);
  double log_nume = pow(x - mean, 2) / (2 * var);

  return log_denom - log_nume;
}

double AnalysisGraph::calculate_delta_log_prior() {
  // If kde of an edge is truely optional ≡ there are some
  // edges without a kde assigned, we should not access it
  // using .value() (In the case of kde being missing, this
  // will throw an exception). We should follow a process
  // similar to Tran_Mat_Cell::sample_from_prior
  KDE& kde = this->graph[this->previous_beta.first].kde;

  // We have to return: log( p( β_new )) - log( p( β_old ))
  return kde.logpdf(this->graph[this->previous_beta.first].beta) -
         kde.logpdf(this->previous_beta.second);
}

void AnalysisGraph::revert_back_to_previous_state() {
  this->log_likelihood = this->previous_log_likelihood;

  this->graph[this->previous_beta.first].beta = this->previous_beta.second;

  // Reset the transition matrix cells that were changed
  // TODO: Can we change the transition matrix only when the sample is
  // accpeted?
  this->update_transition_matrix_cells(this->previous_beta.first);
}

void AnalysisGraph::sample_from_posterior() {
  // Sample a new transition matrix from the proposal distribution
  this->sample_from_proposal();

  double delta_log_prior = this->calculate_delta_log_prior();

  this->set_log_likelihood();
  double delta_log_likelihood =
      this->log_likelihood - this->previous_log_likelihood;

  double delta_log_joint_probability = delta_log_prior + delta_log_likelihood;

  double acceptance_probability = min(1.0, exp(delta_log_joint_probability));

  if (acceptance_probability < this->uni_dist(this->rand_num_generator)) {
    // Reject the sample
    this->revert_back_to_previous_state();
  }
}

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
  try {
    (*this)[concept].add_indicator(indicator, source);
    this->indicators_in_CAG.insert(indicator);
  }
  catch (const out_of_range& oor) {
    error("Error: AnalysisGraph::set_indicator()\n"
          "\tConcept: {0} is not in the CAG\n"
          "\tIndicator: {1} with Source: {2}"
          "\tCannot be added\n",
          concept,
          indicator,
          source);
  }
}

void AnalysisGraph::delete_indicator(string concept, string indicator) {
  try {
    (*this)[concept].delete_indicator(indicator);
    this->indicators_in_CAG.erase(indicator);
  }
  catch (const out_of_range& oor) {
    error("Error: AnalysisGraph::delete_indicator()\n"
          "\tConcept: {0} is not in the CAG\n"
          "\tIndicator: {1} cannot be deleted",
          concept,
          indicator);
  }
}

void AnalysisGraph::delete_all_indicators(string concept) {
  try {
    (*this)[concept].clear_indicators();
  }
  catch (const out_of_range& oor) {
    error("Error: AnalysisGraph::delete_indicator()\n"
          "\tConcept: {} is not in the CAG\n"
          "\tIndicators cannot be deleted",
          concept);
  }
}
