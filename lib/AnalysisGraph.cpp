#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sqlite3.h>
#include <utility>

#include "itertools.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/progress.hpp>

#include "AnalysisGraph.hpp"
#include "data.hpp"
#include "rng.hpp"
#include "tran_mat_cell.hpp"
#include "utils.hpp"
#include <fmt/format.h>

#include <typeinfo>

using namespace std;
using boost::edge;
using boost::source;
using boost::target;
using boost::make_iterator_range;
using fmt::print;

const size_t DEFAULT_N_SAMPLES = 100;

enum InitialBeta { ZERO, ONE, HALF, MEAN, RANDOM };

typedef unordered_map<string, vector<double>> AdjectiveResponseMap;

typedef vector<vector<vector<double>>> ObservedStateSequence;

typedef pair<tuple<string, int, string>, tuple<string, int, string>>
    CausalFragment;

AdjectiveResponseMap
construct_adjective_response_map(size_t n_kernels = DEFAULT_N_SAMPLES) {
  using utils::hasKey;
  sqlite3 *db;
  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);

  if (rc == 1)
    throw "Could not open db\n";

  sqlite3_stmt *stmt;
  const char *query = "select * from gradableAdjectiveData";
  rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);

  AdjectiveResponseMap adjective_response_map;

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    string adjective =
        string(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2)));
    double response = sqlite3_column_double(stmt, 6);
    if (hasKey(adjective_response_map, adjective)) {
      adjective_response_map[adjective] = {response};
    }
    else {
      adjective_response_map[adjective].push_back(response);
    }
  }

  for (auto &[k, v] : adjective_response_map) {
    v = KDE(v).resample(n_kernels);
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return adjective_response_map;
}

/**
 * The AnalysisGraph class is the main model/interface for Delphi.
 */
class AnalysisGraph {
  DiGraph graph;

  public:
  AnalysisGraph() {}
  // Manujinda: I had to move this up since I am usign this within the private:
  // block This is ugly. We need to re-factor the code to make it pretty again
  auto vertices() { return make_iterator_range(boost::vertices(graph)); }

  auto successors(int i) {
    return make_iterator_range(boost::adjacent_vertices(i, graph));
  }

  // Allocate a num_verts x num_verts 2D array (vector of vectors)
  void allocate_A_beta_factors() {
    this->A_beta_factors.clear();

    int num_verts = boost::num_vertices(graph);

    for (int vert = 0; vert < num_verts; ++vert) {
      this->A_beta_factors.push_back(
          vector<shared_ptr<Tran_Mat_Cell>>(num_verts));
    }
  }

  void print_A_beta_factors() {
    int num_verts = boost::num_vertices(graph);

    for (int row = 0; row < num_verts; ++row) {
      for (int col = 0; col < num_verts; ++col) {
        cout << endl
             << "Printing cell: (" << row << ", " << col << ") " << endl;
        if (this->A_beta_factors[row][col]) {
          this->A_beta_factors[row][col]->print_beta2product();
        }
      }
    }
  }

  private:
  // Maps each concept name to the vertex id of the
  // vertex that concept is represented in the CAG
  // concept name --> CAV vertex id
  unordered_map<string, int> name_to_vertex = {};

  // Keeps track of indicators in CAG to ensure there are no duplicates.
  // vector<string> indicators_in_CAG;
  unordered_set<string> indicators_in_CAG;

  // A_beta_factors is a 2D array (vector of vectors) that keeps track
  // of the β factors involved with each cell of the transition matrix A.
  //
  // Accordign to our current model, which uses variables and their partial
  // derivatives with respect to each other ( x --> y, βxy = ∂y/∂x ),
  // atmost half of the transition matrix cells can be affected by βs.
  // According to the way we organize the transition matrix, the cells
  // A[row][col] where row is an even index and col is an odd index
  // are such cells.
  //
  // Each cell of matrix A_beta_factors represent all the directed paths
  // starting at the vertex equal to the column index of the matrix and
  // ending at the vertex equal to the row index of the matrix.
  //
  // Each cell of matrix A_beta_factors is an object of Tran_Mat_Cell class.
  vector<vector<shared_ptr<Tran_Mat_Cell>>> A_beta_factors;

  // A set of (row, column) numbers of the 2D matrix A_beta_factors
  // where the cell (row, column) depends on β factors.
  set<pair<int, int>> beta_dependent_cells;

  // Maps each β to all the transition matrix cells that are dependent on it.
  multimap<pair<int, int>, pair<int, int>> beta2cell;

  double t = 0.0;
  double delta_t = 1.0;
  vector<Eigen::VectorXd> s0;

  // Latent state that is evolved by sampling.
  // Since s0 is used to represent a sequence of latent states,
  // I named this s0_original. Once things are refactored, we might be able to
  // convert this to s0
  Eigen::VectorXd s0_original;

  // Transition matrix that is evolved by sampling.
  // Since variable A has been already used locally in other methods,
  // I chose to name this A_orginal. After refactoring the code, we could
  // rename this to A.
  Eigen::MatrixXd A_original;

  int n_timesteps;
  int pred_timesteps;

  // Accumulates the transition matrices for accepted samples
  // Access: [ sample number ]
  // vector<Eigen::MatrixXd> training_sampled_transition_matrix_sequence;

  // Accumulates the latent states for accepted samples
  // Access this as
  // latent_state_sequences[ sample ][ time step ]
  vector<vector<Eigen::VectorXd>> training_latent_state_sequence_s;

  // This is a column of the
  // this->training_latent_state_sequence_s
  // prediction_initial_latent_state_s.size() = this->res
  // TODO: If we make the code using this variable to directly fetch the values
  // from this->training_latent_state_sequence_s, we can get rid of this
  vector<Eigen::VectorXd> prediction_initial_latent_state_s;
  vector<string> pred_range;

  // Access this as
  // prediction_latent_state_sequence_s[ sample ][ time step ]
  vector<vector<Eigen::VectorXd>> predicted_latent_state_sequence_s;

  // Access this as
  // prediction_observed_state_sequence_s
  //                            [ sample ][ time step ][ vertex ][ indicator ]
  vector<ObservedStateSequence> predicted_observed_state_sequence_s;

  // Sampling resolution. Default is 200
  int res = 200;

  // Training start
  int init_training_year;
  int init_training_month;

  // Keep track whether the model is trained.
  // Used to check whether there is a trained model before calling
  // generate_prediction()
  bool trained = false;

  // Access this as
  // latent_state_sequences[ sample ][ time step ]
  vector<vector<Eigen::VectorXd>> latent_state_sequences;

  // Access this as
  // latent_state_sequence[ time step ]
  vector<Eigen::VectorXd> latent_state_sequence;

  // Access this as
  // observed_state_sequences[ sample ][ time step ][ vertex ][ indicator ]
  vector<ObservedStateSequence> observed_state_sequences;

  // Access this as
  // observed_state_sequence[ time step ][ vertex ][ indicator ]
  ObservedStateSequence observed_state_sequence;

  vector<Eigen::MatrixXd> transition_matrix_collection;

  // Remember the old β and the edge where we perturbed the β.
  // We need this to revert the system to the previous state if the proposal
  // gets rejected.
  pair<boost::graph_traits<DiGraph>::edge_descriptor, double> previous_beta;

  double log_likelihood = 0.0;
  double previous_log_likelihood = 0.0;

  AnalysisGraph(DiGraph G, unordered_map<string, int> name_to_vertex)
      : graph(G), name_to_vertex(name_to_vertex){};

  /**
   * Finds all the simple paths starting at the start vertex and
   * ending at the end vertex.
   * Uses find_all_paths_between_util() as a helper to recursively find the
   * paths
   */
  void find_all_paths_between(int start, int end) {
    // Mark all the vertices are not visited
    for_each(vertices(), [&](int v) { this->graph[v].visited = false; });

    // Create a vector of ints to store paths.
    vector<int> path;

    this->find_all_paths_between_util(start, end, path);
  }

  /**
   * Recursively finds all the simple paths starting at the start vertex and
   * ending at the end vertex. Used by find_all_paths_between()
   * Paths found are added to the Tran_Mat_Cell object that is tracking the
   * transition matrix cell (2*end, 2*start)
   *
   * @param start: Start vertex of the path
   * @param end  : End vertex of the path
   * @param path : A path starting at vettex start that is being explored
   *
   * @return void
   */
  void find_all_paths_between_util(int start, int end, vector<int> &path) {
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
        if (!graph[v].visited) {
          this->find_all_paths_between_util(v, end, path);
        }
      });
    }

    // Remove current vertex from the path and make it unvisited
    path.pop_back();
    graph[start].visited = false;
  };

  /*
   ==========================================================================
   Utilities
   ==========================================================================
  */
  void set_default_initial_state() {
    // Let vertices of the CAG be v = 0, 1, 2, 3, ...
    // Then,
    //    indexes 2*v keeps track of the state of each variable v
    //    indexes 2*v+1 keeps track of the state of ∂v/∂t
    int num_verts = boost::num_vertices(graph);
    int num_els = num_verts * 2;

    this->s0_original = Eigen::VectorXd(num_els);
    this->s0_original.setZero();

    for (int i = 0; i < num_els; i += 2) {
      this->s0_original(i) = 1.0;
    }
  }

  mt19937 rand_num_generator;

  // Uniform distribution used by the MCMC sampler
  uniform_real_distribution<double> uni_dist;

  // Normal distrubution used to perturb β
  normal_distribution<double> norm_dist;

  void initialize_random_number_generator() {
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

  public:
  ~AnalysisGraph() {}

  /**
   * A method to construct an AnalysisGraph object given a JSON-serialized list
   * of INDRA statements.
   *
   * @param filename: The path to the file containing the JSON-serialized INDRA
   * statements.
   */
  static AnalysisGraph from_json_file(string filename,
                                      double belief_score_cutoff = 0.9) {
    using utils::load_json;
    auto json_data = load_json(filename);

    DiGraph G;

    unordered_map<string, int> nameMap = {};

    for (auto stmt : json_data) {
      if (stmt["type"] == "Influence" and
          stmt["belief"] > belief_score_cutoff) {
        auto subj = stmt["subj"]["concept"]["db_refs"]["UN"][0][0];
        auto obj = stmt["obj"]["concept"]["db_refs"]["UN"][0][0];
        if (!subj.is_null() and !obj.is_null()) {

          string subj_str = subj.dump();
          string obj_str = obj.dump();

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

  /**
   * A method to construct an AnalysisGraph object given from a vector of
   * ( subject, object ) pairs (Statements)
   *
   * @param statements: A vector of CausalFragment objects
   */
  static AnalysisGraph
  from_causal_fragments(vector<CausalFragment> causal_fragments) {
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

  auto add_node() { return boost::add_vertex(graph); }

  void update_meta_data() {
    // Update the internal meta-data
    for (int vert_id : vertices()) {
      this->name_to_vertex[this->graph[vert_id].name] = vert_id;
    }

    // Recalculate all the directed simple paths
    this->find_all_paths();
  }

  void remove_node(string concept) {
    auto node_to_remove = this->name_to_vertex.extract(concept);

    if (node_to_remove) // Concept is in the CAG
    {
      // Delete all the edges incident to this node
      boost::clear_vertex(node_to_remove.mapped(), this->graph);

      // Remove the vetex
      boost::remove_vertex(node_to_remove.mapped(), this->graph);

      this->update_meta_data();
    }
    else // indicator_old is not attached to this node
    {
      cerr << "AnalysisGraph::remove_vertex()" << endl;
      cerr << "\tConcept: " << concept << " not present in the CAG!\n" << endl;
    }
  }

  // Note:
  //      Although just calling this->remove_node(concept) within the loop
  //          for( string concept : concept_s )
  //      is suffifient to implement this method, it is not very efficient.
  //      It updates meta-datastructurs and re-calculates directed simple paths
  //      for each vertex removed
  //
  //      Therefore, the code in this->remove_node() has been duplicated with
  //      slightly different flow to achive a more efficient execution.
  void remove_nodes(vector<string> concepts) {
    vector<string> invalid_concept_s;

    for (string concept : concepts) {
      auto node_to_remove = this->name_to_vertex.extract(concept);

      if (node_to_remove) // Concept is in the CAG
      {
        // Delete all the edges incident to this node
        boost::clear_vertex(node_to_remove.mapped(), this->graph);

        // Remove the vetex
        boost::remove_vertex(node_to_remove.mapped(), this->graph);
      }
      else // indicator_old is not attached to this node
      {
        invalid_concept_s.push_back(concept);
      }
    }

    if (invalid_concept_s.size() < concepts.size()) {
      // Some concepts have been removed
      // Update the internal meta-data
      this->update_meta_data();
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

  auto add_edge(int i, int j) { boost::add_edge(i, j, graph); }

  //auto add_edge(string source, string target) {
    //boost::add_edge(
        //this->name_to_vertex[source], this->name_to_vertex[target], graph);
  //}

  void remove_edge(string src, string tgt) {
    int src_id = -1;
    int tgt_id = -1;

    try {
      src_id = this->name_to_vertex.at(src);
    }
    catch (const out_of_range &oor) {
      cerr << "AnalysisGraph::remove_edge" << endl;
      cerr << "\tSource vertex " << src << " is not in the CAG!" << endl;
      return;
    }

    try {
      tgt_id = this->name_to_vertex.at(tgt);
    }
    catch (const out_of_range &oor) {
      cerr << "AnalysisGraph::remove_edge" << endl;
      cerr << "\tTarget vertex " << tgt << " is not in the CAG!" << endl;
      return;
    }

    pair<int, int> edge = make_pair(src_id, tgt_id);

    // edge ≡ β
    if (this->beta2cell.find(edge) == this->beta2cell.end()) {
      cerr << "AnalysisGraph::remove_edge" << endl;
      cerr << "\tThere is no edge from " << src << " to " << tgt
           << " in the CAG!" << endl;
      return;
    }

    // Remove the edge
    boost::remove_edge(src_id, tgt_id, this->graph);

    // Recalculate all the directed simple paths
    this->find_all_paths();
  }

  auto edges() { return make_iterator_range(boost::edges(graph)); }

  auto predecessors(int i) {
    return make_iterator_range(boost::inv_adjacent_vertices(i, graph));
  }

  auto predecessors(string node_name) {
    return predecessors(this->name_to_vertex[node_name]);
  }

  // Merge node n1 into node n2, with the option to specify relative polarity.
  //void merge_nodes(string n1, string n2, bool same_polarity = true) {
    //for (auto p : predecessors(n1)) {
      //auto e = boost::edge(p, this->name_to_vertex[n1], this->graph).first;
      //if (!same_polarity) {
        //for (Statement s : this->graph[edge].evidence) {
          //s.object.polarity = -s.object.polarity;
        //}
      //}

      ////auto [edge, is_new_edge] =
          ////boost::add_edge(p, this->name_to_vertex[n2], this->graph);
      ////if (!is_new_edge) {
        ////for (auto s : this->graph[e].evidence) {
          ////this->graph[edge].evidence.push_back(s);
        ////}
      //}
    //}
  //}

  auto out_edges(int i) {
    return make_iterator_range(boost::out_edges(i, graph));
  }

  double get_beta(string source_vertex_name, string target_vertex_name) {
    // This is ∂target / ∂source
    return this->A_original(2 * this->name_to_vertex[target_vertex_name],
                            2 * this->name_to_vertex[source_vertex_name] + 1);
  }

  void construct_beta_pdfs() {
    using utils::get;
    using utils::lmap;

    double sigma_X = 1.0;
    double sigma_Y = 1.0;
    auto adjective_response_map = construct_adjective_response_map();
    vector<double> marginalized_responses;
    for (auto [adjective, responses] : adjective_response_map) {
      for (auto response : responses) {
        marginalized_responses.push_back(response);
      }
    }

    marginalized_responses =
        KDE(marginalized_responses).resample(DEFAULT_N_SAMPLES);

    for (auto e : edges()) {
      vector<double> all_thetas = {};

      for (Statement stmt : graph[e].evidence) {
        Event subject = stmt.subject;
        Event object = stmt.object;

        string subj_adjective = subject.adjective;
        string obj_adjective = object.adjective;

        auto subj_responses = lmap([&](auto x) { return x * subject.polarity; },
                                   get(adjective_response_map,
                                       subj_adjective,
                                       marginalized_responses));

        auto obj_responses = lmap(
            [&](auto x) { return x * object.polarity; },
            get(adjective_response_map, obj_adjective, marginalized_responses));

        for (auto [x, y] : iter::product(subj_responses, obj_responses)) {
          all_thetas.push_back(atan2(sigma_Y * y, sigma_X * x));
        }
      }

      // TODO: Why kde is optional in struct Edge?
      // It seems all the edges get assigned with a kde
      graph[e].kde = KDE(all_thetas);

      // Initialize the initial β for this edge
      // TODO: Decide the correct way to initialize this
      graph[e].beta = graph[e].kde.value().mu;
    }
  }

  /*
   * Find all the simple paths between all the paris of nodes of the graph
   */
  void find_all_paths() {
    auto verts = vertices();

    // Allocate the 2D array that keeps track of the cells of the transition
    // matrix (A_original) that are dependent on βs.
    // This function can be called anytime after creating the CAG.
    this->allocate_A_beta_factors();

    // Clear the multimap that keeps track of cells in the transition matrix
    // that are dependent on each β.
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
    int num_verts = boost::num_vertices(graph);

    for (int row = 0; row < num_verts; ++row) {
      for (int col = 0; col < num_verts; ++col) {
        if (this->A_beta_factors[row][col]) {
          this->A_beta_factors[row][col]->allocate_datastructures();
        }
      }
    }
  }

  /*
   * Prints the simple paths found between all pairs of nodes of the graph
   * Groupd according to the starting and ending vertex.
   * find_all_paths() should be called before this to populate the paths
   */
  void print_all_paths() {
    int num_verts = boost::num_vertices(graph);

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
  void print_cells_affected_by_beta(int source, int target) {
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

  /*
   ==========================================================================
   Sampling and inference
   ----------------------

   This section contains code for sampling and Bayesian inference.
   ==========================================================================
  */

  // Sample elements of the stochastic transition matrix from the
  // prior distribution, based on gradable adjectives.
  void sample_initial_transition_matrix_from_prior() {
    int num_verts = boost::num_vertices(this->graph);

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
    for (auto &[row, col] : this->beta_dependent_cells) {
      this->A_original(row * 2, col * 2 + 1) =
          // this->A_beta_factors[row][col]->sample_from_prior(this->graph);
          this->A_beta_factors[row][col]->compute_cell(this->graph);
    }
  }

  /**
   * Utility function that converts a time range given a start date and end date
   * into an integer value.
   * At the moment returns the number of months withing the time range.
   * This should be the number of traing data time points we have
   *
   * @param start_year  : Start year of the training data sequence
   * @param start_month : Start month of the training data sequence
   * @param end_year    : End year of the training data sequence
   * @param end_month   : End month of the training data sequence
   *
   * @return            : Number of months in the training data sequence
   *                      Including both start and end months
   */
  int calculate_num_timesteps(int start_year,
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

  /**
   * Get the observed state (values for all the indicators)
   * for a given time point from data.
   * See data.hpp::get_data_value() for missing data rules.
   * Note: units are automatically set according
   * to the parameterization of the given CAG.
   *
   * @param year    : Year of the time point data is extracted
   * @param month   : Month of the time point data is extracted
   * @param country : Country where the data is about
   * @param state   : State where the data is about
   *
   * @return        : Observed state vector for the specified location
   *                  on the specified time point.
   *                  Access it as: [ vertex id ][ indicator id ]
   */
  vector<vector<double>> get_observed_state_from_data(
      int year, int month, string country = "South Sudan", string state = "") {
    int num_verts = boost::num_vertices(this->graph);

    // Access
    // [ vertex ][ indicator ]
    vector<vector<double>> observed_state(num_verts);

    for (int v = 0; v < num_verts; v++) {
      vector<Indicator> &indicators = this->graph[v].indicators;

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

  /**
   * Set the observed state sequence for a given time range from data.
   * The sequence includes both ends of the range.
   * See data.hpp::get_data_value() for missing data rules.
   * Note: units are automatically set according
   * to the parameterization of the given CAG.
   *
   * @param start_year  : Start year of the sequence of data
   * @param start_month : Start month of the sequence of data
   * @param end_year    : End year of the sequence of data
   * @param end_month   : End month of the sequenec of data
   * @param country     : Country where the data is about
   * @param state       : State where the data is about
   *
   */
  void set_observed_state_sequence_from_data(int start_year,
                                             int start_month,
                                             int end_year,
                                             int end_month,
                                             string country = "South Sudan",
                                             string state = "") {
    this->observed_state_sequence.clear();

    // Access
    // [ timestep ][ veretx ][ indicator ]
    this->observed_state_sequence = ObservedStateSequence(this->n_timesteps);

    int year = start_year;
    int month = start_month;

    for (int ts = 0; ts < this->n_timesteps; ts++) {
      this->observed_state_sequence[ts] =
          get_observed_state_from_data(year, month, country, state);

      if (month == 12) {
        year++;
        month = 1;
      }
      else {
        month++;
      }
    }
  }

  /**
   * Utility function that sets an initial latent state from observed data.
   * This is used for the inference of the transition matrix as well as the
   * training latent state sequences.
   *
   * @param timestep: Optional setting for setting the initial state to be other
   *                  than the first time step. Not currently used.
   *                  0 <= timestep < this->n_timesteps
   */
  void set_initial_latent_state_from_observed_state_sequence(int timestep = 0) {
    int num_verts = boost::num_vertices(this->graph);

    this->set_default_initial_state();

    for (int v = 0; v < num_verts; v++) {
      vector<Indicator> &indicators = this->graph[v].indicators;

      for (int i = 0; i < indicators.size(); i++) {
        Indicator &ind = indicators[i];

        double ind_mean = ind.get_mean();

        while (ind_mean == 0) {
          ind_mean = this->norm_dist(this->rand_num_generator);
        }

        double ind_value = this->observed_state_sequence[timestep][v][i];

        // TODO: If ind_mean is very close to zero, this could overflow
        // Even indexes of the latent state vector represent variables
        this->s0_original(2 * v) = ind_value / ind_mean;

        if (timestep == this->n_timesteps - 1) {
          double prev_ind_value =
              this->observed_state_sequence[timestep - 1][v][i];
          double prev_state_value = prev_ind_value / ind_mean;
          double diff = this->s0_original(2 * v) - prev_state_value;
          // TODO: Why this is different from else branch?
          this->s0_original(2 * v + 1) =
              this->norm_dist(this->rand_num_generator) + diff;
        }
        else {
          double next_ind_value =
              this->observed_state_sequence[timestep + 1][v][i];
          double next_state_value = next_ind_value / ind_mean;
          double diff = next_state_value - this->s0_original(2 * v);
          // TODO: Why this is different from if branch?
          this->s0_original(2 * v + 1) = diff;
        }
      }
    }
  }

  void set_random_initial_latent_state() {
    int num_verts = boost::num_vertices(this->graph);

    this->set_default_initial_state();

    for (int v = 0; v < num_verts; v++) {
      this->s0_original(2 * v + 1) =
          0.1 * this->uni_dist(this->rand_num_generator);
    }
  }

  /**
   * To help experiment with initializing βs to differet values
   *
   * @param ib: Criteria to initialize β
   */
  void init_betas_to(InitialBeta ib = InitialBeta::MEAN) {
    switch (ib) {
    // Initialize the initial β for this edge
    // Note: I am repeating the loop within each case for efficiency.
    // If we embed the switch withn the for loop, there will be less code
    // but we will evaluate the switch for each iteration through the loop
    case InitialBeta::ZERO:
      for (boost::graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
        graph[e].beta = 0;
      }
      break;
    case InitialBeta::ONE:
      for (boost::graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
        graph[e].beta = 1.0;
      }
      break;
    case InitialBeta::HALF:
      for (boost::graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
        graph[e].beta = 0.5;
      }
      break;
    case InitialBeta::MEAN:
      for (boost::graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
        graph[e].beta = graph[e].kde.value().mu;
      }
      break;
    case InitialBeta::RANDOM:
      for (boost::graph_traits<DiGraph>::edge_descriptor e : this->edges()) {
        // this->uni_dist() gives a random number in range [0, 1]
        // Multiplying by 2 scales the range to [0, 2]
        // Sustracting 1 moves the range to [-1, 1]
        graph[e].beta = this->uni_dist(this->rand_num_generator) * 2 - 1;
      }
      break;
    }
  }

  /**
   * Train a prediction model given a CAG with indicators
   *
   * @param start_year  : Start year of the sequence of data
   * @param start_month : Start month of the sequence of data
   * @param end_year    : End year of the sequence of data
   * @param end_month   : End month of the sequenec of data
   * @param res         : Sampling resolution. The number of samples to retain.
   * @param burn        : Number of samples to throw away. Start retaining
   *                      samples after throwing away this many samples.
   * @param country     : Country where the data is about
   * @param state       : State where the data is about
   * @param units       : Units for each indicator. Maps
   *                      indicator name --> unit
   * @param initial_beta: Criteria to initialize β
   *
   */
  void train_model(int start_year = 2012,
                   int start_month = 1,
                   int end_year = 2017,
                   int end_month = 12,
                   int res = 200,
                   int burn = 10000,
                   string country = "South Sudan",
                   string state = "",
                   map<string, string> units = {},
                   InitialBeta initial_beta = InitialBeta::ZERO) {
    this->n_timesteps = this->calculate_num_timesteps(
        start_year, start_month, end_year, end_month);
    this->res = res;
    this->init_betas_to(initial_beta);
    this->sample_initial_transition_matrix_from_prior();
    this->parameterize(country, state, start_year, start_month, units);

    this->init_training_year = start_year;
    this->init_training_month = start_month;

    if (!syntheitc_data_experiment) {
      this->set_observed_state_sequence_from_data(
          start_year, start_month, end_year, end_month, country, state);
    }

    this->set_initial_latent_state_from_observed_state_sequence();

    this->set_log_likelihood();

    // Accumulates the transition matrices for accepted samples
    // Access: [ sample number ]
    // training_sampled_transition_matrix_sequence.clear();
    // training_sampled_transition_matrix_sequence =
    //    vector<Eigen::MatrixXd>(this->res);
    //
    // generate_prediction()      uses
    // sample_from_likelihood. It uses
    // transition_mateix_collection
    // So to keep things simple for the moment
    // I had to fall back to
    // transition_matrix_collection
    // HOWEVER: The purpose of transition_matrix_collection
    // seem to be different in the prevous code than here.
    // In the earlier code, (in sample_from_prior()) this is
    // populated with DEFAULT_N_SAMPLES of initial transition matrices.
    // Here we populate it with res number of sampler emitted transition
    // matrices.
    //
    this->transition_matrix_collection.clear();
    this->transition_matrix_collection = vector<Eigen::MatrixXd>(this->res);

    // Accumulates the latent states for accepted samples
    // Access this as
    // latent_state_sequences[ sample ][ time step ]
    this->training_latent_state_sequence_s.clear();
    this->training_latent_state_sequence_s =
        vector<vector<Eigen::VectorXd>>(this->res);

    boost::progress_display progress_bar(burn + this->res);
    boost::progress_timer t;

    for (int _ = 0; _ < burn; _++) {
      this->sample_from_posterior();
      ++progress_bar;
    }

    for (int samp = 0; samp < this->res; samp++) {
      this->sample_from_posterior();
      // this->training_sampled_transition_matrix_sequence[samp] =
      this->transition_matrix_collection[samp] = this->A_original;
      this->training_latent_state_sequence_s[samp] =
          this->latent_state_sequence;
      ++progress_bar;
    }

    this->trained = true;
    fmt::print("\n");
    return;
  }

  /**
   * Utility function that sets initial latent states for predictions.
   * During model training a latent state sequence is inferred for each sampled
   * transition matrix, these are then used as the initial states for
   * predictions.
   *
   * @param timestep: Optional setting for setting the initial state to be other
                      than the first time step. Ensures that the correct
                      initial state is used.
                      0 <= timestep < this->n_timesteps
   */
  void set_initial_latent_states_for_prediction(int timestep = 0) {
    this->prediction_initial_latent_state_s.clear();
    this->prediction_initial_latent_state_s =
        vector<Eigen::VectorXd>(this->res);

    transform(
        this->training_latent_state_sequence_s.begin(),
        this->training_latent_state_sequence_s.end(),
        this->prediction_initial_latent_state_s.begin(),
        [&timestep](vector<Eigen::VectorXd> &ls) { return ls[timestep]; });
  }

  /**
   * Sample a collection of observed state sequences from the likelihood
   * model given a collection of transition matrices.
   *
   * @param timesteps: The number of timesteps for the sequences.
   */
  void sample_predicted_latent_state_sequence_s_from_likelihood(int timesteps) {
    this->n_timesteps = timesteps;

    int num_verts = boost::num_vertices(this->graph);

    // Allocate memory for prediction_latent_state_sequence_s
    this->predicted_latent_state_sequence_s.clear();
    this->predicted_latent_state_sequence_s = vector<vector<Eigen::VectorXd>>(
        this->res,
        vector<Eigen::VectorXd>(this->n_timesteps,
                                Eigen::VectorXd(num_verts * 2)));

    for (int samp = 0; samp < this->res; samp++) {
      this->predicted_latent_state_sequence_s[samp][0] =
          this->prediction_initial_latent_state_s[samp];

      for (int ts = 1; ts < this->n_timesteps; ts++) {
        this->predicted_latent_state_sequence_s[samp][ts] =
            this->transition_matrix_collection[samp] *
            this->predicted_latent_state_sequence_s[samp][ts - 1];
      }
    }
  }

  /** Generate predicted observed state sequenes given predicted latent state
   * sequences using the emission model
   */
  void
  generate_predicted_observed_state_sequence_s_from_predicted_latent_state_sequence_s() {
    // Allocate memory for observed_state_sequences
    this->predicted_observed_state_sequence_s.clear();
    this->predicted_observed_state_sequence_s = vector<ObservedStateSequence>(
        this->res,
        ObservedStateSequence(this->n_timesteps, vector<vector<double>>()));

    for (int samp = 0; samp < this->res; samp++) {
      vector<Eigen::VectorXd> &sample =
          this->predicted_latent_state_sequence_s[samp];

      transform(sample.begin(),
                sample.end(),
                this->predicted_observed_state_sequence_s[samp].begin(),
                [this](Eigen::VectorXd latent_state) {
                  return this->sample_observed_state(latent_state);
                });
    }
  }

  /**
   * Given a trained model, generate this->res number of
   * predicted observed state sequences.
   *
   * @param start_year  : Start year of the prediction
   *                      Should be >= the start year of training
   * @param start_month : Start month of the prediction
   *                      If training and prediction start years are equal
   *                      should be >= the start month of training
   * @param end_year    : End year of the prediction
   * @param end_month   : End month of the prediction
   *
   * @return Predicted observed state (indicator value) sequence for the
   *         prediction period including start and end time points.
   *         This is a tuple.
   *         The first element is a vector of strings with lables for each
   *         time point predicted (year-month).
   *         The second element contains predicted values. Access it as:
   *         [ sample number ][ time point ][ vertex name ][ indicator name ]
   */
  pair<vector<string>,
       vector<vector<unordered_map<string, unordered_map<string, double>>>>>
  generate_prediction(int start_year,
                      int start_month,
                      int end_year,
                      int end_month) {
    if (!this->trained) {
      fmt::print("Passed untrained Causal Analysis Graph (CAG) Model. \n",
                 "Try calling <CAG>.train_model(...) first!");
      throw "Model not yet trained";
    }

    if (start_year < this->init_training_year ||
        (start_year == this->init_training_year &&
         start_month < this->init_training_month)) {
      fmt::print("The initial prediction date can't be before the\n"
                 "inital training date. Defaulting initial prediction date\n"
                 "to initial training date.");
      start_year = this->init_training_year;
      start_month = this->init_training_month;
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
        this->calculate_num_timesteps(this->init_training_year,
                                      this->init_training_month,
                                      end_year,
                                      end_month);

    this->pred_timesteps = this->calculate_num_timesteps(
        start_year, start_month, end_year, end_month);

    int diff_timesteps = total_timesteps - this->pred_timesteps;
    int truncate = 0;

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

    int pred_init_timestep = diff_timesteps;

    if (diff_timesteps >= this->n_timesteps) {
      /*
       *              total_timesteps
       *   ____________________________________________
       *  |                                            |
       *  v                                            v
       *tr_st           tr_ed     pr_st              pr_ed
       *  |---------------|---------|------------------|
       *  ^               ^         ^                  ^
       *  |_______________|_________|__________________|
       *  :  n_timesteps   truncate :   pred_timesteps
       *  ^                         ^
       *  |_________________________|
       *            diff
       */
      pred_init_timestep = this->n_timesteps - 1;
      truncate = diff_timesteps - this->n_timesteps;
      this->pred_timesteps += truncate; // = total_timesteps - this->n_timesteps
    }

    this->set_initial_latent_states_for_prediction(pred_init_timestep);

    this->sample_predicted_latent_state_sequence_s_from_likelihood(
        this->pred_timesteps);
    this->generate_predicted_observed_state_sequence_s_from_predicted_latent_state_sequence_s();

    for (vector<Eigen::VectorXd> &latent_state_s :
         this->predicted_latent_state_sequence_s) {
      latent_state_s.erase(latent_state_s.begin(),
                           latent_state_s.begin() + truncate);
    }

    for (ObservedStateSequence &observed_state_s :
         this->predicted_observed_state_sequence_s) {
      observed_state_s.erase(observed_state_s.begin(),
                             observed_state_s.begin() + truncate);
    }

    this->pred_timesteps -= truncate;
    return make_pair(this->pred_range, this->format_prediction_result());
  }

  /**
   * Format the prediction result into a format python callers favor.
   *
   * @param pred_timestes: Number of timesteps in the predicted sequence.
   *
   * @return Re-formatted prediction result.
   *         Access it as:
   *         [ sample number ][ time point ][ vertex name ][ indicator name ]
   */
  vector<vector<unordered_map<string, unordered_map<string, double>>>>
  format_prediction_result() {
    // Access
    // [ sample ][ time_step ][ vertex_name ][ indicator_name ]
    vector<vector<unordered_map<string, unordered_map<string, double>>>>
        result = vector<
            vector<unordered_map<string, unordered_map<string, double>>>>(
            this->res,
            vector<unordered_map<string, unordered_map<string, double>>>(
                this->pred_timesteps));

    for (int samp = 0; samp < this->res; samp++) {
      for (int ts = 0; ts < this->pred_timesteps; ts++) {
        for (auto [vert_name, vert_id] : this->name_to_vertex) {
          for (auto [ind_name, ind_id] : this->graph[vert_id].indicator_names) {
            result[samp][ts][vert_name][ind_name] =
                this->predicted_observed_state_sequence_s[samp][ts][vert_id]
                                                         [ind_id];
          }
        }
      }
    }

    return result;
  }

  /**
   * this->generate_prediction() must be called before callign this method.
   * Outputs raw predictions for a given indicator that were generated by
   * generate_prediction(). Each column is a time step and the rows are the
   * samples for that time step.
   *
   * @param indicator: A string representing the indicator variable for which we
   *                   want predictions for.

   * @return A this->res x this->pred_timesteps dimension 2D array
   *         (vector of vectors)
   *
  */
  vector<vector<double>> prediction_to_array(string indicator) {
    int vert_id = -1;
    int ind_id = -1;

    vector<vector<double>> result =
        vector<vector<double>>(this->res, vector<double>(this->pred_timesteps));

    // Find the vertex id the indicator is attached to and
    // the indicator id of it.
    // TODO: We can make this more efficient by making indicators_in_CAG
    // a map from indicator names to vertices they are attached to.
    // This is just a quick and dirty implementation
    for (auto [v_name, v_id] : this->name_to_vertex) {
      for (auto [i_name, i_id] : this->graph[v_id].indicator_names) {
        if (indicator.compare(i_name) == 0) {
          vert_id = v_id;
          ind_id = i_id;
          goto indicator_found;
        }
      }
    }
    // Program will reach hear only if the indicator is not found
    fmt::print("AnalysisGraph::prediction_to_array - indicator {} not found!\n",
               indicator);
    throw IndicatorNotFoundException(fmt::format(
        "AnalysisGraph::prediction_to_array - indicator \"{}\" not found!\n",
        indicator));

  indicator_found:

    for (int samp = 0; samp < this->res; samp++) {
      for (int ts = 0; ts < this->pred_timesteps; ts++) {
        result[samp][ts] =
            this->predicted_observed_state_sequence_s[samp][ts][vert_id]
                                                     [ind_id];
      }
    }

    return result;
  }

  vector<Eigen::VectorXd> synthetic_latent_state_sequence;
  // ObservedStateSequence synthetic_observed_state_sequence;
  bool syntheitc_data_experiment = false;

  void generate_synthetic_latent_state_sequence_from_likelihood() {
    int num_verts = boost::num_vertices(this->graph);

    // Allocate memory for synthetic_latent_state_sequence
    this->synthetic_latent_state_sequence.clear();
    this->synthetic_latent_state_sequence = vector<Eigen::VectorXd>(
        this->n_timesteps, Eigen::VectorXd(num_verts * 2));

    this->synthetic_latent_state_sequence[0] = this->s0_original;

    for (int ts = 1; ts < this->n_timesteps; ts++) {
      this->synthetic_latent_state_sequence[ts] =
          this->A_original * this->synthetic_latent_state_sequence[ts - 1];
    }
  }

  void
  generate_synthetic_observed_state_sequence_from_synthetic_latent_state_sequence() {
    // Allocate memory for observed_state_sequences
    this->observed_state_sequence.clear();
    this->observed_state_sequence =
        ObservedStateSequence(this->n_timesteps, vector<vector<double>>());

    transform(this->synthetic_latent_state_sequence.begin(),
              this->synthetic_latent_state_sequence.end(),
              this->observed_state_sequence.begin(),
              [this](Eigen::VectorXd latent_state) {
                return this->sample_observed_state(latent_state);
              });
  }

  pair<ObservedStateSequence,
       pair<vector<string>,
            vector<
                vector<unordered_map<string, unordered_map<string, double>>>>>>
  test_inference_with_synthetic_data(
      int start_year = 2015,
      int start_month = 1,
      int end_year = 2015,
      int end_month = 12,
      int res = 100,
      int burn = 900,
      string country = "South Sudan",
      string state = "",
      map<string, string> units = {},
      InitialBeta initial_beta = InitialBeta::HALF) {

    syntheitc_data_experiment = true;

    this->n_timesteps = this->calculate_num_timesteps(
        start_year, start_month, end_year, end_month);
    this->init_betas_to(initial_beta);
    this->sample_initial_transition_matrix_from_prior();
    cout << this->A_original << endl;
    this->parameterize(country, state, start_year, start_month, units);

    // Initialize the latent state vector at time 0
    this->set_random_initial_latent_state();
    this->generate_synthetic_latent_state_sequence_from_likelihood();
    this->generate_synthetic_observed_state_sequence_from_synthetic_latent_state_sequence();

    for (vector<vector<double>> obs : this->observed_state_sequence) {
      fmt::print("({}, {})\n", obs[0][0], obs[1][0]);
    }

    this->train_model(start_year,
                      start_month,
                      end_year,
                      end_month,
                      res,
                      burn,
                      country,
                      state,
                      units,
                      InitialBeta::ZERO);

    return make_pair(this->observed_state_sequence,
                     this->generate_prediction(
                         start_year, start_month, end_year, end_month));

    syntheitc_data_experiment = false;
  }

  // TODO: Need testing
  /**
   * Sample observed state vector.
   * This is the implementation of the emission function.
   *
   * @param latent_state: Latent state vector.
   *                      This has 2 * number of vertices in the CAG.
   *                      Even indices track the state of each vertex.
   *                      Odd indices track the state of the derivative.
   *
   * @return Observed state vector. Observed state for each indicator for each
   * vertex.
   *         Indexed by: [ vertex id ][ indicator id ]
   */
  vector<vector<double>> sample_observed_state(Eigen::VectorXd latent_state) {
    int num_verts = boost::num_vertices(this->graph);

    assert(num_verts == latent_state.size() / 2);

    vector<vector<double>> observed_state(num_verts);

    for (int v = 0; v < num_verts; v++) {
      vector<Indicator> &indicators = this->graph[v].indicators;

      observed_state[v] = vector<double>(indicators.size());

      // Sample observed value of each indicator around the mean of the
      // indicator
      // scaled by the value of the latent state that caused this observation.
      // TODO: Question - Is ind.mean * latent_state[ 2*v ] correct?
      //                  Shouldn't it be ind.mean + latent_state[ 2*v ]?
      transform(indicators.begin(),
                indicators.end(),
                observed_state[v].begin(),
                [&](Indicator ind) {
                  normal_distribution<double> gaussian(
                      ind.mean * latent_state[2 * v], ind.stdev);

                  return gaussian(this->rand_num_generator);
                });
    }

    return observed_state;
  }

  /**
   * Find all the transition matrix (A) cells that are dependent on the β
   * attached to the provided edge and update them.
   * Acts upon this->A_original
   *
   * @param e: The directed edge ≡ β that has been perturbed
   */
  void update_transition_matrix_cells(
      boost::graph_traits<DiGraph>::edge_descriptor e) {
    pair<int, int> beta =
        make_pair(source(e, this->graph), target(e, this->graph));

    typedef multimap<pair<int, int>, pair<int, int>>::iterator MMAPIterator;

    pair<MMAPIterator, MMAPIterator> beta_dept_cells =
        this->beta2cell.equal_range(beta);

    // TODO: I am introducing this to implement calculate_Δ_log_prior
    // Remember the cells of A that got changed and their previous values
    // this->A_cells_changed.clear();

    for (MMAPIterator it = beta_dept_cells.first; it != beta_dept_cells.second;
         it++) {
      int &row = it->second.first;
      int &col = it->second.second;

      // Note that I am remembering row and col instead of 2*row and 2*col+1
      // row and col resembles an edge in the CAG: row -> col
      // ( 2*row, 2*col+1 ) is the transition mateix cell that got changed.
      // this->A_cells_changed.push_back( make_tuple( row, col, A( row * 2, col
      // * 2 + 1 )));

      this->A_original(row * 2, col * 2 + 1) =
          this->A_beta_factors[row][col]->compute_cell(this->graph);
    }
  }

  /**
   * Sample a new transition matrix from the proposal distribution,
   * given a current candidate transition matrix.
   * In practice, this amounts to:
   *    Selecting a random β.
   *    Perturbing it a bit.
   *    Updating all the transition matrix cells that are dependent on it.
   */
  // TODO: Need testng
  // TODO: Before calling sample_from_proposal() we must call
  // AnalysisGraph::find_all_paths()
  // TODO: Before calling sample_from_proposal(), we mush assign initial βs and
  // run Tran_Mat_Cell::compute_cell() to initialize the first transistion
  // matrix.
  // TODO: Update Tran_Mat_Cell::compute_cell() to calculate the proper value.
  // At the moment it just computes sum of length of all the paths realted to
  // this cell
  void sample_from_proposal() {
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

  void set_latent_state_sequence() {
    int num_verts = boost::num_vertices(this->graph);

    // Allocate memory for latent_state_sequence
    this->latent_state_sequence.clear();
    this->latent_state_sequence = vector<Eigen::VectorXd>(this->n_timesteps);

    this->latent_state_sequence[0] = this->s0_original;

    for (int ts = 1; ts < this->n_timesteps; ts++) {
      this->latent_state_sequence[ts] =
          this->A_original * this->latent_state_sequence[ts - 1];
    }
  }

  double log_normpdf(double x, double mean, double sd) {
    double var = pow(sd, 2);
    double log_denom = -0.5 * log(2 * M_PI) - log(sd);
    double log_nume = pow(x - mean, 2) / (2 * var);

    return log_denom - log_nume;
  }

  void set_log_likelihood() {
    this->previous_log_likelihood = this->log_likelihood;
    this->log_likelihood = 0.0;

    this->set_latent_state_sequence();

    for (int ts = 0; ts < this->n_timesteps; ts++) {
      const Eigen::VectorXd &latent_state = this->latent_state_sequence[ts];

      // Access
      // observed_state[ vertex ][ indicator ]
      const vector<vector<double>> &observed_state =
          this->observed_state_sequence[ts];

      for (int v : vertices()) {
        const int &num_inds_for_v = observed_state[v].size();

        for (int i = 0; i < observed_state[v].size(); i++) {
          const double &value = observed_state[v][i];
          const Indicator &ind = graph[v].indicators[i];

          // Even indices of latent_state keeps track of the state of each
          // vertex
          double log_likelihood_component = this->log_normpdf(
              value, latent_state[2 * v] * ind.mean, ind.stdev);
          this->log_likelihood += log_likelihood_component;
        }
      }
    }
  }

  double calculate_delta_log_prior() {
    // If kde of an edge is truely optional ≡ there are some
    // edges without a kde assigned, we should not access it
    // using .value() (In the case of kde being missing, this
    // will throw an exception). We should follow a process
    // similar to Tran_Mat_Cell::sample_from_prior
    KDE &kde = this->graph[this->previous_beta.first].kde.value();

    // We have to return: log( p( β_new )) - log( p( β_old ))
    return kde.logpdf(this->graph[this->previous_beta.first].beta) -
           kde.logpdf(this->previous_beta.second);
  }

  void revert_back_to_previous_state() {
    this->log_likelihood = this->previous_log_likelihood;

    this->graph[this->previous_beta.first].beta = this->previous_beta.second;

    // Reset the transition matrix cells that were changed
    // TODO: Can we change the transition matrix only when the sample is
    // accpeted?
    this->update_transition_matrix_cells(this->previous_beta.first);
  }

  /**
   * Run Bayesian inference - sample from the posterior distribution.
   */
  void sample_from_posterior() {
    // Sample a new transition matrix from the proposal distribution
    this->sample_from_proposal();

    double delta_log_prior = this->calculate_delta_log_prior();

    this->set_log_likelihood();
    double delta_log_likelihood =
        this->log_likelihood - this->previous_log_likelihood;

    double delta_log_joint_probability = delta_log_prior + delta_log_likelihood;

    double acceptance_probability = min(1.0, exp(delta_log_joint_probability));

    if (acceptance_probability < uni_dist(this->rand_num_generator)) {
      // Reject the sample
      this->revert_back_to_previous_state();
    }
  }

  // ==========================================================================
  // Manipulation
  // ==========================================================================

  void set_indicator(string concept, string indicator, string source) {
    if (this->indicators_in_CAG.find(indicator) !=
        this->indicators_in_CAG.end()) {
      print("{0} already exists in Casual Analysis Graph, Indicator {0} was "
            "not added to Concept {1}.",
            indicator,
            concept);
      return;
    }
    try {
      this->graph[this->name_to_vertex.at(concept)].add_indicator(indicator,
                                                                  source);
      this->indicators_in_CAG.insert(indicator);
    }
    catch (const out_of_range &oor) {
      cerr << "Error: AnalysisGraph::set_indicator()\n"
           << "\tConcept: " << concept << " is not in the CAG\n";
      cerr << "\tIndicator: " << indicator << " with Source: " << source
           << endl;
      cerr << "\tCannot be added\n";
    }
  }

  /*
  // TODO: Demosntrate how to use the Node::get_indicator() method
  // with the custom exception.
  // Not sure whether we need this method in AnalaysisGraph
  // so that python side can directly access the Indicator class
  // objects and maipulate them (we need to fiture out how to map a
  // custom class from C++ into python for this) - harder
  // or
  // mirror getter and setter methods of the Indicator class
  // in AnalysisGraph and make the python side call them - easier.
  Indicator get_indicator(string concept, string indicator) {
    try {
      return graph[name_to_vertex.at(concept)].get_indicator(indicator);
    } catch (const out_of_range &oor) {
      print("Error: AnalysisGraph::get_indicator()\n");
      print("\tConcept: {} is not in the CAG\n", concept);
    } catch (IndicatorNotFoundException &infe) {
      cerr << "Error: AnalysisGraph::get_indicator()\n"
                << "\tindicator: " << infe.what()
                << " is not attached to CAG node " << concept << endl;
    }
  }
  */

  void replace_indicator(string concept,
                         string indicator_old,
                         string indicator_new,
                         string source) {

    if (this->indicators_in_CAG.find(indicator_new) !=
        this->indicators_in_CAG.end()) {
      print("{0} already exists in Casual Analysis Graph, Indicator {0} did "
            "not replace Indicator {1} for Concept {2}.",
            indicator_new,
            indicator_old,
            concept);
      return;
    }

    try {
      this->graph[this->name_to_vertex.at(concept)].replace_indicator(
          indicator_old, indicator_new, source);
      this->indicators_in_CAG.insert(indicator_new);
      this->indicators_in_CAG.erase(indicator_old);
    }
    catch (const out_of_range &oor) {
      cerr << "Error: AnalysisGraph::replace_indicator()\n"
           << "\tConcept: " << concept << " is not in the CAG\n";
      cerr << "\tIndicator: " << indicator_old << " cannot be replaced" << endl;
    }
  }

  /*
    ==========================================================================
    Model parameterization
    *Loren: I am going to try to port this, I'll try not to touch anything up
    top
    and only push changes that compile. If I do push a change that breaks things
    you could probably just comment out this section.*
    ==========================================================================
  */

  /**
   * Map each concept node in the AnalysisGraph instance to one or more
   * tangible quantities, known as 'indicators'.
   *
   * @param n: Int representing number of indicators to attach per node.
   * Default is 1 since our model so far is configured for only 1 indicator per
   * node.
   */
  void map_concepts_to_indicators(int n = 1) {
    sqlite3 *db;
    int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
    if (rc) {
      print("Could not open db\n");
      return;
    }
    sqlite3_stmt *stmt;
    string query_base =
        "select Source, Indicator from concept_to_indicator_mapping ";
    string query;
    for (int v : this->vertices()) {
      query = query_base + "where `Concept` like " + "'" + this->graph[v].name +
              "'";
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
                reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
            ind_name = string(
                reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1)));
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

  /**
   * Parameterize the indicators of the AnalysisGraph..
   *
   */
  void parameterize(string country = "South Sudan",
                    string state = "",
                    int year = 2012,
                    int month = 1,
                    map<string, string> units = {}) {
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

  auto print_nodes() {
    print("Vertex IDs and their names in the CAG\n");
    print("Vertex ID : Name\n");
    print("--------- : ----\n");
    for_each(vertices(), [&](auto v) {
      cout << v << "         : " << this->graph[v].name << endl;
    });
  }

  auto print_edges() {
    for_each(edges(), [&](auto e) {
      cout << "(" << source(e, graph) << ", " << target(e, graph) << ")"
           << endl;
    });
  }

  void print_name_to_vertex() {
    for (auto [name, vert] : this->name_to_vertex) {
      cout << name << " -> " << vert << endl;
    }
    cout << endl;
  }

  auto to_dot() {
    using boost::make_label_writer;
    using boost::write_graphviz;

    write_graphviz(
        cout, graph, make_label_writer(boost::get(&Node::name, graph)));
  }

  auto print_indicators() {
    for (int v : this->vertices()) {
      cout << "node " << v << ": " << this->graph[v].name << ":" << endl;
      for (auto [name, vert] : this->graph[v].indicator_names) {
        cout << "\t"
             << "indicator " << vert << ": " << name << endl;
      }
    }
  }
};
