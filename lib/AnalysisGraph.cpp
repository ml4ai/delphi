#include "AnalysisGraph.hpp"
#include "data.hpp"
#include <boost/range/algorithm/for_each.hpp>
#include <cmath>
#include <sqlite3.h>

using namespace std;

auto AnalysisGraph::vertices() {
  return boost::make_iterator_range(boost::vertices(this->graph));
}
auto AnalysisGraph::successors(int i) {
  return boost::make_iterator_range(boost::adjacent_vertices(i, this->graph));
}

auto AnalysisGraph::successors(string node_name) {
  return boost::make_iterator_range(boost::adjacent_vertices(
      this->name_to_vertex.at(node_name), this->graph));
}

void AnalysisGraph::initialize_random_number_generator() {
  // Define the random number generator
  // All the places we need random numbers, share this generator

  this->rand_num_generator = RNG::rng()->get_RNG();

  // Uniform distribution used by the MCMC sampler
  this->uni_dist = uniform_real_distribution<double>(0.0, 1.0);

  // Normal distrubution used to perturb β
  this->norm_dist = normal_distribution<double>(0.0, 1.0);

  this->construct_beta_pdfs();
  this->find_all_paths();
}

void AnalysisGraph::parameterize(string country,
                                 string state,
                                 int year,
                                 int month,
                                 map<string, string> units) {
  double stdev;
  for (int v : this->vertices()) {
    for (auto [name, i] : this->graph[v].indicator_names) {
      if (units.find(name) != units.end()) {
        this->graph[v].indicators[i].set_unit(units[name]);
        this->graph[v].indicators[i].set_mean(
            get_data_value(name, country, state, year, month, units[name]));
        stdev = 0.1 * abs(this->graph[v].indicators[i].get_mean());
        this->graph[v].indicators[i].set_stdev(stdev);
      }
      else {
        this->graph[v].indicators[i].set_default_unit();
        this->graph[v].indicators[i].set_mean(
            get_data_value(name, country, state, year, month));
        stdev = 0.1 * abs(this->graph[v].indicators[i].get_mean());
        this->graph[v].indicators[i].set_stdev(stdev);
      }
    }
  }
}

void AnalysisGraph::allocate_A_beta_factors() {
  this->A_beta_factors.clear();

  int num_verts = boost::num_vertices(this->graph);

  for (int vert = 0; vert < num_verts; ++vert) {
    this->A_beta_factors.push_back(
        vector<shared_ptr<Tran_Mat_Cell>>(num_verts));
  }
}

void AnalysisGraph::print_A_beta_factors() {
  int num_verts = boost::num_vertices(this->graph);

  for (int row = 0; row < num_verts; ++row) {
    for (int col = 0; col < num_verts; ++col) {
      cout << endl << "Printing cell: (" << row << ", " << col << ") " << endl;
      if (this->A_beta_factors[row][col]) {
        this->A_beta_factors[row][col]->print_beta2product();
      }
    }
  }
}
void AnalysisGraph::find_all_paths_between(int start, int end) {
  // Mark all the vertices are not visited
  boost::for_each(vertices(), [&](int v) { this->graph[v].visited = false; });

  // Create a vector of ints to store paths.
  vector<int> path;

  this->find_all_paths_between_util(start, end, path);
}
void AnalysisGraph::find_all_paths_between_util(int start,
                                                int end,
                                                vector<int>& path) {
  // Mark the current vertex visited
  this->graph[start].visited = true;

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
  else {
    // Current vertex is not the destination
    // Recursively process all the vertices adjacent to the current vertex
    for_each(successors(start), [&](int v) {
      if (!this->graph[v].visited) {
        this->find_all_paths_between_util(v, end, path);
      }
    });
  }

  // Remove current vertex from the path and make it unvisited
  path.pop_back();
  this->graph[start].visited = false;
};
void AnalysisGraph::set_default_initial_state() {
  // Let vertices of the CAG be v = 0, 1, 2, 3, ...
  // Then,
  //    indexes 2*v keeps track of the state of each variable v
  //    indexes 2*v+1 keeps track of the state of ∂v/∂t
  int num_verts = boost::num_vertices(this->graph);
  int num_els = num_verts * 2;

  this->s0_original = Eigen::VectorXd(num_els);
  this->s0_original.setZero();

  for (int i = 0; i < num_els; i += 2) {
    this->s0_original(i) = 1.0;
  }
}
int AnalysisGraph::get_vertex_id_for_concept(string concept, string caller) {
  int vert_id = -1;

  try {
    vert_id = this->name_to_vertex.at(concept);
  }
  catch (const out_of_range& oor) {
    cerr << "AnalysisGraph::" << caller << endl;
    cerr << "ERROR:" << endl;
    cerr << "\tThe concept " << concept << " is not in the CAG!" << endl;
    rethrow_exception(current_exception());
  }

  return vert_id;
}
int AnalysisGraph::get_degree(int vertex_id) {
  return boost::in_degree(vertex_id, this->graph) +
         boost::out_degree(vertex_id, this->graph);
}
void AnalysisGraph::remove_node(int node_id) {
  // Delete all the edges incident to this node
  boost::clear_vertex(node_id, this->graph);

  // Remove the vetex
  boost::remove_vertex(node_id, this->graph);

  // Update the internal meta-data
  for (int vert_id : vertices()) {
    this->name_to_vertex[this->graph[vert_id].name] = vert_id;
  }

}

AnalysisGraph AnalysisGraph::from_json_file(string filename,
                                            double belief_score_cutoff) {
  using utils::load_json;
  auto json_data = load_json(filename);

  DiGraph G;

  unordered_map<string, int> nameMap = {};

  for (auto stmt : json_data) {
    if (stmt["type"] == "Influence" and stmt["belief"] > belief_score_cutoff) {
      auto subj = stmt["subj"]["concept"]["db_refs"]["UN"][0][0];
      auto obj = stmt["obj"]["concept"]["db_refs"]["UN"][0][0];
      if (!subj.is_null() and !obj.is_null()) {

        string subj_str = subj.get<std::string>();
        string obj_str = obj.get<std::string>();

        // Add the nodes to the graph if they are not in it already
        for (string name : {subj_str, obj_str}) {
          if (nameMap.find(name) == nameMap.end()) {
            int v = boost::add_vertex(G);
            nameMap[name] = v;
            G[v].name = name;
          }
        }

        // Add the edge to the graph if it is not in it already
        auto [e, exists] =
            boost::add_edge(nameMap[subj_str], nameMap[obj_str], G);
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

          if(subj_polarity.is_null()){
            subj_polarity = 1;
          }
          if(obj_polarity.is_null()){
            obj_polarity = 1;
          }

          Event subject{subj_adjective, subj_polarity, ""};
          Event object{obj_adjective, obj_polarity, ""};
          G[e].evidence.push_back(Statement{subject, object});
        }
      }
    }
  }
  AnalysisGraph ag = AnalysisGraph(G, nameMap);
  ag.initialize_random_number_generator();
  return ag;
}

AnalysisGraph AnalysisGraph::get_subgraph_for_concept(string concept,
                                                      int depth,
                                                      bool inward) {

  int vert_id =
      get_vertex_id_for_concept(concept, "get_subgraph_for_concept()");

  int num_verts = boost::num_vertices(graph);

  unordered_set<int> vertices_to_keep = unordered_set<int>();
  unordered_set<string> vertices_to_remove;

  if (inward) {
    // All paths of length less than or equal to depth ending at vert_id
    for (int col = 0; col < num_verts; ++col) {
      if (this->A_beta_factors[vert_id][col]) {
        unordered_set<int> vwh =
            this->A_beta_factors[vert_id][col]->get_vertices_within_hops(depth,
                                                                         false);

        vertices_to_keep.insert(vwh.begin(), vwh.end());
      }
    }
  }
  else {
    // All paths of length less than or equal to depth beginning at vert_id
    for (int row = 0; row < num_verts; ++row) {
      if (this->A_beta_factors[row][vert_id]) {
        unordered_set<int> vwh =
            this->A_beta_factors[row][vert_id]->get_vertices_within_hops(depth,
                                                                         true);

        vertices_to_keep.insert(vwh.begin(), vwh.end());
      }
    }
  }

  if (vertices_to_keep.size() == 0) {
    cerr << "AnalysisGraph::get_subgraph_for_concept()" << endl;
    cerr << "WARNING:" << endl;
    cerr << "\tReturning and empty CAG!" << endl;
  }

  // Determine the vertices to be removed
  for (int vert_id : vertices()) {
    if (vertices_to_keep.find(vert_id) == vertices_to_keep.end()) {
      vertices_to_remove.insert(this->graph[vert_id].name);
    }
  }

  // Make a copy of current AnalysisGraph
  // TODO: We have to make sure that we are making a deep copy.
  //       Test so far does not show suspicious behavior
  AnalysisGraph G_sub = *this;
  G_sub.remove_nodes(vertices_to_remove);
  G_sub.find_all_paths();

  return G_sub;
}
AnalysisGraph AnalysisGraph::get_subgraph_for_concept_pair(
    string source_concept, string target_concept, int cutoff) {

  int src_id = get_vertex_id_for_concept(source_concept,
                                         "get_subgraph_for_concept_pair()");
  int tgt_id = get_vertex_id_for_concept(target_concept,
                                         "get_subgraph_for_concept_pair()");

  unordered_set<int> vertices_to_keep;
  unordered_set<string> vertices_to_remove;

  // All paths of length less than or equal to depth ending at vert_id
  if (this->A_beta_factors[tgt_id][src_id]) {
    vertices_to_keep =
        this->A_beta_factors[tgt_id][src_id]
            ->get_vertices_on_paths_shorter_than_or_equal_to(cutoff);

    if (vertices_to_keep.size() == 0) {
      cerr << "AnalysisGraph::get_subgraph_for_concept_pair()" << endl;
      cerr << "WARNING:" << endl;
      cerr << "\tThere are no paths of length <= " << cutoff
           << " from source concept " << source_concept
           << " --to-> target concept " << target_concept << endl;
      cerr << "\tReturning and empty CAG!" << endl;
    }
  }
  else {
    cerr << "AnalysisGraph::get_subgraph_for_concept_pair()" << endl;
    cerr << "WARNING:" << endl;
    cerr << "\tThere are no paths from source concept " << source_concept
         << " --to-> target concept " << target_concept << endl;
    cerr << "\tReturning and empty CAG!" << endl;
  }

  // Determine the vertices to be removed
  for (int vert_id : vertices()) {
    if (vertices_to_keep.find(vert_id) == vertices_to_keep.end()) {
      vertices_to_remove.insert(this->graph[vert_id].name);
    }
  }

  // Make a copy of current AnalysisGraph
  // TODO: We have to make sure that we are making a deep copy.
  //       Test so far does not show suspicious behavior
  AnalysisGraph G_sub = *this;
  G_sub.remove_nodes(vertices_to_remove);
  G_sub.find_all_paths();

  return G_sub;
}
void AnalysisGraph::prune(int cutoff) {
  int num_verts = boost::num_vertices(this->graph);
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
        if (this->beta2cell.find(edge) != this->beta2cell.end()) {
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
void AnalysisGraph::remove_node(string concept) {
  auto node_to_remove = this->name_to_vertex.extract(concept);

  if (node_to_remove) // Concept is in the CAG
  {
    // Note: This is an overlaoded private method that takes in a vertex id
    this->remove_node(node_to_remove.mapped());

    // Recalculate all the directed simple paths
    this->find_all_paths();
  }
  else // indicator_old is not attached to this node
  {
    cerr << "AnalysisGraph::remove_vertex()" << endl;
    cerr << "\tConcept: " << concept << " not present in the CAG!\n" << endl;
  }
}
void AnalysisGraph::remove_nodes(unordered_set<string> concepts) {
  vector<string> invalid_concept_s;

  for (string concept : concepts) {
    auto node_to_remove = this->name_to_vertex.extract(concept);

    if (node_to_remove) // Concept is in the CAG
    {
      // Note: This is an overlaoded private method that takes in a vertex id
      this->remove_node(node_to_remove.mapped());
    }
    else // Concept is not in the CAG
    {
      invalid_concept_s.push_back(concept);
    }
  }

  if (invalid_concept_s.size() < concepts.size()) {
    // Some concepts have been removed
    // Recalculate all the directed simple paths
    this->find_all_paths();
  }

  if (invalid_concept_s.size() > 0) {
    // There were some invalid concepts
    cerr << "AnalysisGraph::remove_vertex()" << endl;
    cerr << "\tFollowing concepts were not present in the CAG!" << endl;
    for (string invalid_concept : invalid_concept_s) {
      cerr << "\t\t" << invalid_concept << endl;
    }
    cerr << endl;
  }
}
void AnalysisGraph::remove_edge(string src, string tgt) {
  int src_id = -1;
  int tgt_id = -1;

  try {
    src_id = this->name_to_vertex.at(src);
  }
  catch (const out_of_range& oor) {
    cerr << "AnalysisGraph::remove_edge" << endl;
    cerr << "\tSource vertex " << src << " is not in the CAG!" << endl;
    return;
  }

  try {
    tgt_id = this->name_to_vertex.at(tgt);
  }
  catch (const out_of_range& oor) {
    cerr << "AnalysisGraph::remove_edge" << endl;
    cerr << "\tTarget vertex " << tgt << " is not in the CAG!" << endl;
    return;
  }

  pair<int, int> edge = make_pair(src_id, tgt_id);

  // edge ≡ β
  if (this->beta2cell.find(edge) == this->beta2cell.end()) {
    cerr << "AnalysisGraph::remove_edge" << endl;
    cerr << "\tThere is no edge from " << src << " to " << tgt << " in the CAG!"
         << endl;
    return;
  }

  // Remove the edge
  boost::remove_edge(src_id, tgt_id, this->graph);

  // Recalculate all the directed simple paths
  this->find_all_paths();
}
void AnalysisGraph::remove_edges(vector<pair<string, string>> edges) {

  vector<pair<int, int>> edge_id_s = vector<pair<int, int>>(edges.size());

  set<string> invalid_src_s;
  set<string> invalid_tgt_s;
  set<pair<string, string>> invalid_edg_s;

  transform(edges.begin(),
            edges.end(),
            edge_id_s.begin(),
            [this](pair<string, string> edg) {
              int src_id;
              int tgt_id;

              // Flag invalid source vertices
              try {
                src_id = this->name_to_vertex.at(edg.first);
              }
              catch (const out_of_range& oor) {
                src_id = -1;
              }

              // Flag invalid target vertices
              try {
                tgt_id = this->name_to_vertex.at(edg.second);
              }
              catch (const out_of_range& oor) {
                tgt_id = -1;
              }

              // Flag invalid edges
              if (src_id != -1 && tgt_id != -1) {
                pair<int, int> edg_id = make_pair(src_id, tgt_id);

                if (this->beta2cell.find(edg_id) == this->beta2cell.end()) {
                  src_id = -2;
                }
              }

              return make_pair(src_id, tgt_id);
            });

  bool has_invalid_src_s = false;
  bool has_invalid_tgt_s = false;
  bool has_invalid_edg_s = false;

  for (int e = 0; e < edge_id_s.size(); e++) {
    bool valid_edge = true;

    if (edge_id_s[e].first == -1) {
      invalid_src_s.insert(edges[e].first);
      valid_edge = false;
      has_invalid_src_s = true;
    }

    if (edge_id_s[e].second == -1) {
      invalid_tgt_s.insert(edges[e].second);
      valid_edge = false;
      has_invalid_tgt_s = true;
    }

    if (edge_id_s[e].first == -2) {
      invalid_edg_s.insert(edges[e]);
      valid_edge = false;
      has_invalid_edg_s = true;
    }

    if (valid_edge) {
      // Remove the edge
      boost::remove_edge(edge_id_s[e].first, edge_id_s[e].second, this->graph);
    }
  }

  if (has_invalid_src_s || has_invalid_tgt_s || has_invalid_edg_s) {
    cerr << "ERROR: AnalysisGraph::remove_edges" << endl;

    if (has_invalid_src_s) {
      cerr << "\tFollowing source vertexes are not in the CAG!" << endl;
      for (string invalid_src : invalid_src_s) {
        cerr << "\t\t" << invalid_src << endl;
      }
    }

    if (has_invalid_tgt_s) {
      cerr << "\tFollowing target vertexes are not in the CAG!" << endl;
      for (string invalid_tgt : invalid_tgt_s) {
        cerr << "\t\t" << invalid_tgt << endl;
      }
    }

    if (has_invalid_edg_s) {
      cerr << "\tFollowing edges are not in the CAG!" << endl;
      for (pair<string, string> invalid_edg : invalid_edg_s) {
        cerr << "\t\t" << invalid_edg.first << " --to-> " << invalid_edg.second
             << endl;
      }
    }
  }

  // Recalculate all the directed simple paths
  this->find_all_paths();
}
pair<Agraph_t*, GVC_t*> AnalysisGraph::to_agraph() {
  using delphi::gv::set_property, delphi::gv::add_node;
  Agraph_t* G = agopen(const_cast<char*>("G"), Agdirected, NULL);
  GVC_t* gvc;
  gvc = gvContext();

  // Set global properties
  set_property(G, AGNODE, "shape", "rectangle");
  set_property(G, AGNODE, "style", "rounded");
  set_property(G, AGNODE, "color", "maroon");
  set_property(G, AGNODE, "fontname", "Helvetica");

  Agnode_t* src;
  Agnode_t* trgt;
  Agedge_t* edge;

  // Add CAG links
  for (auto e : edges()) {
    string source_name = this->graph[boost::source(e, this->graph)].name;
    string target_name = this->graph[boost::target(e, this->graph)].name;

    src = add_node(G, source_name);
    set_property(src, "label", source_name);

    trgt = add_node(G, target_name);
    set_property(trgt, "label", target_name);

    edge = agedge(G, src, trgt, 0, true);
  }

  // Add concepts, indicators, and link them.
  for (auto v : vertices()) {
    string concept_name = this->graph[v].name;
    for (auto indicator : this->graph[v].indicators) {
      src = add_node(G, concept_name);
      trgt = add_node(G, indicator.name);
      set_property(trgt, "label", indicator.name);
      set_property(trgt, "style", "rounded,filled");
      set_property(trgt, "fillcolor", "lightblue");

      edge = agedge(G, src, trgt, 0, true);
    }
  }
  gvLayout(gvc, G, "dot");
  return make_pair(G, gvc);
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

void AnalysisGraph::to_png(string filename) {
  auto [G, gvc] = this->to_agraph();
  gvRenderFilename(gvc, G, "png", const_cast<char*>(filename.c_str()));
  gvFreeLayout(gvc, G);
  agclose(G);
  gvFreeContext(gvc);
}
void AnalysisGraph::print_indicators() {
  for (int v : this->vertices()) {
    cout << "node " << v << ": " << this->graph[v].name << ":" << endl;
    for (auto [name, vert] : this->graph[v].indicator_names) {
      cout << "\t"
           << "indicator " << vert << ": " << name << endl;
    }
  }
}

AnalysisGraph
AnalysisGraph::from_causal_fragments(vector<CausalFragment> causal_fragments) {
  DiGraph G;

  unordered_map<string, int> nameMap = {};

  for (CausalFragment cf : causal_fragments) {
    Event subject = Event(cf.first);
    Event object = Event(cf.second);

    string subj_name = subject.concept_name;
    string obj_name = object.concept_name;

    // Add the nodes to the graph if they are not in it already
    for (string name : {subj_name, obj_name}) {
      if (nameMap.find(name) == nameMap.end()) {
        int v = boost::add_vertex(G);
        nameMap[name] = v;
        G[v].name = name;
      }
    }

    // Add the edge to the graph if it is not in it already
    auto [e, exists] =
        boost::add_edge(nameMap[subj_name], nameMap[obj_name], G);

    G[e].evidence.push_back(Statement{subject, object});
  }
  AnalysisGraph ag = AnalysisGraph(G, nameMap);
  ag.initialize_random_number_generator();
  return ag;
}
void AnalysisGraph::merge_nodes_old(string n1, string n2, bool same_polarity) {
  for (auto predecessor : predecessors(n1)) {
    auto e =
        boost::edge(predecessor, this->name_to_vertex[n1], this->graph).first;
    if (!same_polarity) {
      for (Statement stmt : this->graph[e].evidence) {
        stmt.object.polarity = -stmt.object.polarity;
      }
    }

    auto [edge, is_new_edge] =
        boost::add_edge(predecessor, this->name_to_vertex[n2], this->graph);
    for (auto s : this->graph[e].evidence) {
      this->graph[edge].evidence.push_back(s);
    }
  }

  for (auto successor : successors(n1)) {
    auto e =
        boost::edge(this->name_to_vertex[n1], successor, this->graph).first;
    if (!same_polarity) {
      for (Statement stmt : this->graph[e].evidence) {
        stmt.subject.polarity = -stmt.subject.polarity;
      }
    }

    auto [edge, is_new_edge] =
        boost::add_edge(this->name_to_vertex[n2], successor, this->graph);
    for (auto stmt : this->graph[e].evidence) {
      this->graph[edge].evidence.push_back(stmt);
    }
  }
  remove_node(n1);
}

void AnalysisGraph::merge_nodes(string concept_1,
                                string concept_2,
                                bool same_polarity) {
  int vertex_remove = get_vertex_id_for_concept(concept_1, "merge_nodes()");
  int vertex_keep = get_vertex_id_for_concept(concept_2, "merge_nodes()");

  /*
  int c1_id = get_vertex_id_for_concept(concept_1, "merge_nodes()");
  int c2_id = get_vertex_id_for_concept(concept_2, "merge_nodes()");

  // Choose the node with the higher degree to keep and the other to delete
  int c1_degree = this->get_degree(c1_id);
  int c2_degree = this->get_degree(c2_id);

  int vertex_keep = c1_id;
  int vertex_remove = c2_id;

  if( c1_degree < c2_degree )
  {
    vertex_keep = c2_id;
    vertex_remove = c1_id;
  }
  */

  for (int predecessor : predecessors(vertex_remove)) {

    // Get the edge descripter for
    //                   predecessor --> vertex_remove
    auto edg_remove =
        boost::edge(predecessor, vertex_remove, this->graph).first;

    if (!same_polarity) {
      for (Statement stmt : this->graph[edg_remove].evidence) {
        stmt.object.polarity = -stmt.object.polarity;
      }
    }

    // Add the edge   predecessor --> vertex_keep
    auto [edg_keep, is_new_edge] =
        boost::add_edge(predecessor, vertex_keep, this->graph);

    // Move all the evidence from vertex_delete to the
    // newly created (or existing) edge
    // predecessor --> vertex_keep
    vector<Statement>& evidence_keep = this->graph[edg_keep].evidence;
    vector<Statement>& evidence_move = this->graph[edg_remove].evidence;

    evidence_keep.resize(evidence_keep.size() + evidence_move.size());

    move(evidence_move.begin(),
         evidence_move.end(),
         evidence_keep.end() - evidence_move.size());
  }

  for (int successor : successors(vertex_remove)) {

    // Get the edge descripter for
    //                   vertex_remove --> successor
    auto edg_remove = boost::edge(vertex_remove, successor, this->graph).first;

    if (!same_polarity) {
      for (Statement stmt : this->graph[edg_remove].evidence) {
        stmt.object.polarity = -stmt.object.polarity;
      }
    }

    // Add the edge   successor --> vertex_keep
    auto [edg_keep, is_new_edge] =
        boost::add_edge(vertex_keep, successor, this->graph);

    // Move all the evidence from vertex_delete to the
    // newly created (or existing) edge
    // vertex_keep --> successor
    vector<Statement>& evidence_keep = this->graph[edg_keep].evidence;
    vector<Statement>& evidence_move = this->graph[edg_remove].evidence;

    evidence_keep.resize(evidence_keep.size() + evidence_move.size());

    move(evidence_move.begin(),
         evidence_move.end(),
         evidence_keep.end() - evidence_move.size());
  }

  // Remove vertex_remove from the CAG
  // Note: This is an overlaoded private method that takes in a vertex id
  this->remove_node(vertex_remove);

  // Recalculate all the directed simple paths
  this->find_all_paths();
}
void AnalysisGraph::print_nodes() {
  fmt::print("Vertex IDs and their names in the CAG\n");
  fmt::print("Vertex ID : Name\n");
  fmt::print("--------- : ----\n");
  boost::for_each(vertices(), [&](auto v) {
    cout << v << "         : " << this->graph[v].name << endl;
  });
}
void AnalysisGraph::map_concepts_to_indicators(int n) {
  sqlite3* db;
  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
  if (rc) {
    fmt::print("Could not open db\n");
    return;
  }
  sqlite3_stmt* stmt;
  string query_base =
      "select Source, Indicator from concept_to_indicator_mapping ";
  string query;
  for (int v : this->vertices()) {
    query =
        query_base + "where `Concept` like " + "'" + this->graph[v].name + "'";
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    this->graph[v].clear_indicators();
    bool ind_not_found = false;
    for (int c = 0; c < n; c = c + 1) {
      string ind_source;
      string ind_name;
      do {
        rc = sqlite3_step(stmt);
        if (rc == SQLITE_ROW) {
          ind_source = string(
              reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
          ind_name = string(
              reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
        }
        else {
          ind_not_found = true;
          break;
        }
      } while (this->indicators_in_CAG.find(ind_name) !=
               this->indicators_in_CAG.end());

      if (!ind_not_found) {
        this->graph[v].add_indicator(ind_name, ind_source);
        this->indicators_in_CAG.insert(ind_name);
      }
      else {
        cout << "No more indicators were found, only " << c
             << "indicators attached to " << this->graph[v].name << endl;
        break;
      }
    }
    sqlite3_finalize(stmt);
  }
  sqlite3_close(db);
}
void AnalysisGraph::set_log_likelihood() {
  this->previous_log_likelihood = this->log_likelihood;
  this->log_likelihood = 0.0;

  this->set_latent_state_sequence();

  for (int ts = 0; ts < this->n_timesteps; ts++) {
    const Eigen::VectorXd& latent_state = this->latent_state_sequence[ts];

    // Access
    // observed_state[ vertex ][ indicator ]
    const vector<vector<double>>& observed_state =
        this->observed_state_sequence[ts];

    for (int v : vertices()) {
      const int& num_inds_for_v = observed_state[v].size();

      for (int i = 0; i < observed_state[v].size(); i++) {
        const double& value = observed_state[v][i];
        const Indicator& ind = graph[v].indicators[i];

        // Even indices of latent_state keeps track of the state of each
        // vertex
        double log_likelihood_component =
            this->log_normpdf(value, latent_state[2 * v] * ind.mean, ind.stdev);
        this->log_likelihood += log_likelihood_component;
      }
    }
  }
}
void AnalysisGraph::find_all_paths() {
  auto verts = vertices();

  // Allocate the 2D array that keeps track of the cells of the transition
  // matrix (A_original) that are dependent on βs.
  // This function can be called anytime after creating the CAG.
  this->allocate_A_beta_factors();

  // Clear the multimap that keeps track of cells in the transition
  // matrix that are dependent on each β.
  this->beta2cell.clear();

  // Clear the set of all the β dependent cells
  this->beta_dependent_cells.clear();

  boost::for_each(verts, [&](int start) {
    boost::for_each(verts, [&](int end) {
      if (start != end) {
        this->find_all_paths_between(start, end);
      }
    });
  });

  // Allocate the cell value calculation data structures
  int num_verts = boost::num_vertices(graph);

  for (int row = 0; row < num_verts; ++row) {
    for (int col = 0; col < num_verts; ++col) {
      if (this->A_beta_factors[row][col]) {
        this->A_beta_factors[row][col]->allocate_datastructures();
      }
    }
  }
}

void AnalysisGraph::print_edges() {
  boost::for_each(edges(), [&](auto e) {
    cout << "(" << boost::source(e, this->graph) << ", "
         << boost::target(e, this->graph) << ")" << endl;
  });
}

void AnalysisGraph::print_name_to_vertex() {
  for (auto [name, vert] : this->name_to_vertex) {
    cout << name << " -> " << vert << endl;
  }
  cout << endl;
}

vector<vector<double>> AnalysisGraph::get_observed_state_from_data(
    int year, int month, string country, string state) {
  int num_verts = boost::num_vertices(this->graph);

  // Access
  // [ vertex ][ indicator ]
  vector<vector<double>> observed_state(num_verts);

  for (int v = 0; v < num_verts; v++) {
    vector<Indicator>& indicators = this->graph[v].indicators;

    observed_state[v] = vector<double>(indicators.size(), 0.0);

    transform(
        indicators.begin(),
        indicators.end(),
        observed_state[v].begin(),
        [&](Indicator ind) {
          // get_data_value() is defined in data.hpp
          return get_data_value(
              ind.get_name(), country, state, year, month, ind.get_unit());
        });
  }

  return observed_state;
}

void AnalysisGraph::add_node(string concept) {
  if (this->name_to_vertex.find(concept) == this->name_to_vertex.end()) {
    int v = boost::add_vertex(this->graph);
    this->name_to_vertex[concept] = v;
    this->graph[v].name = concept;
  }
  else {
    fmt::print("AnalysisGraph::add_node()\n\tconcept {} already exists!\n",
               concept);
  }
}

void AnalysisGraph::add_edge(CausalFragment causal_fragment) {
  Event subject = Event(causal_fragment.first);
  Event object = Event(causal_fragment.second);

  string subj_name = subject.concept_name;
  string obj_name = object.concept_name;

  // Add the nodes to the graph if they are not in it already
  this->add_node(subj_name);
  this->add_node(obj_name);

  // Add the edge to the graph if it is not in it already
  auto [e, exists] = boost::add_edge(this->name_to_vertex[subj_name],
                                     this->name_to_vertex[obj_name],
                                     this->graph);

  this->graph[e].evidence.push_back(Statement{subject, object});
}

void AnalysisGraph::change_polarity_of_edge(string source_concept,
                                            int source_polarity,
                                            string target_concept,
                                            int target_polarity) {
  int src_id = this->get_vertex_id_for_concept(source_concept,
                                               "change_polarity_of_edge");
  int tgt_id = this->get_vertex_id_for_concept(target_concept,
                                               "change_polarity_of_edge");

  pair<int, int> edg = make_pair(src_id, tgt_id);

  // edge ≡ β
  if (this->beta2cell.find(edg) != this->beta2cell.end()) {
    // There is a edge from src_concept to tgt_concept
    // get that edge object
    auto e = boost::edge(src_id, tgt_id, this->graph).first;

    this->graph[e].change_polarity(source_polarity, target_polarity);
  }
}
