#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sqlite3.h>
#include <utility>

#include "cppitertools/itertools.hpp"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>

#include "AnalysisGraph.hpp"
#include "utils.hpp"

using std::cout, std::endl, std::unordered_map, std::pair, std::string,
    std::ifstream, std::stringstream, std::map, boost::inner_product,
    boost::edge, boost::source, boost::target, boost::graph_bundle,
    boost::make_label_writer, boost::write_graphviz, boost::lambda::make_const,
    utils::load_json, utils::hasKey, utils::get;

template <class V, class Iterable> vector<V> list(Iterable xs) {
  vector<V> xs_copy;
  for (auto x : xs) {
    xs_copy.push_back(x);
  }
  return xs_copy;
}

template <class F, class V> vector<V> lmap(F f, vector<V> vec) {
  vector<V> transformed_vector;
  for (V x : vec) {
    transformed_vector.push_back(f(x));
  }
  return transformed_vector;
}

const size_t default_n_samples = 100;

unordered_map<string, vector<double>>
construct_adjective_response_map(size_t n_kernels = default_n_samples) {
  sqlite3 *db;
  int rc = sqlite3_open(std::getenv("DELPHI_DB"), &db);
  if (!rc)
    print("Opened db successfully");
  else
    print("Could not open db");

  sqlite3_stmt *stmt;
  const char *query = "select * from gradableAdjectiveData";
  rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
  unordered_map<string, vector<double>> adjective_response_map;

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    string adjective = std::string(
        reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2)));
    double response = sqlite3_column_double(stmt, 6);
    if (hasKey(adjective_response_map, adjective)) {
      adjective_response_map[adjective] = {response};
    } else {
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
  // Manujinda: I had to move this up since I am usign this within the private:
  // block This is ugly. We need to re-factor the code to make it pretty again
  auto vertices() { return boost::make_iterator_range(boost::vertices(graph)); }

  auto successors(int i) {
    return boost::make_iterator_range(boost::adjacent_vertices(i, graph));
  }

private:
  AnalysisGraph(DiGraph G) : graph(G){};

  /**
   * Finds all the simple paths starting at the start vertex and
   * ending at the end vertex.
   * Paths found are appended to the influenced_by data structure in the Node
   * Uses all_paths_between_uitl() as a helper to recursively find the paths
   */
  void all_paths_between(int start, int end) {
    // Mark all the vertices are not visited
    for_each(vertices(), [&](int v) { graph[v].visited = false; });

    // Create a vector of pairs to store paths.
    // A path is stored as a sequrence of edges
    // An edge is a pair: (start, end)
    vector<pair<int, int>> path;

    all_paths_between_util(start, end, path);
  }

  /**
   * Used by all_paths_between()
   * Recursively finds all the simple paths starting at the start vertex and
   * ending at the end vertex.
   * Paths found are appended to the influenced_by data structure in the Node
   */
  void
  all_paths_between_util(int start, int end, vector<pair<int, int>> &path) {
    // Mark the current vertex visited
    graph[start].visited = true;

    // If current vertex is the destination vertex, then
    //   we have found one path. Append that to the end node
    if (start == end) {
      // Append this part to the vertex end's influenced_by map
      // with the first vertex of this path (path[0].first) as the key.
      // The value is a vector of vector of pairs.
      graph[end].influenced_by[path[0].first].push_back(path);
    } else { // Current vertex is not the destination
      // Process all the vertices adjacent to the current node
      for_each(successors(start), [&](int v) {
        if (!graph[v].visited) {
          // Appedn the edge start---v to the path
          path.push_back(pair<int, int>(start, v));

          all_paths_between_util(v, end, path);
        }
      });
    }

    // Remove current vertex from the path and make it unvisited
    path.pop_back();
    graph[start].visited = false;
  };

public:
  /**
   * A method to construct an AnalysisGraph object given a JSON-serialized list
   * of INDRA statements.
   *
   * @param filename The path to the file containing the JSON-serialized INDRA
   * statements.
   */
  static AnalysisGraph from_json_file(string filename) {
    auto json_data = load_json(filename);

    DiGraph G;
    std::unordered_map<string, int> nameMap = {};
    int i = 0;
    for (auto stmt : json_data) {
      if (stmt["type"] == "Influence" and stmt["belief"] > 0.9) {
        auto subj = stmt["subj"]["concept"]["db_refs"]["UN"][0][0];
        auto obj = stmt["obj"]["concept"]["db_refs"]["UN"][0][0];
        if (!subj.is_null() and !obj.is_null()) {

          auto subj_str = subj.dump();
          auto obj_str = obj.dump();

          // Add the nodes to the graph if they are not in it already
          for (auto c : {subj_str, obj_str}) {
            if (nameMap.count(c) == 0) {
              nameMap[c] = i;
              auto v = boost::add_vertex(G);
              G[v].name = c;
              i++;
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
            G[e].causalFragments.push_back(CausalFragment{
                subj_adjective, obj_adjective, subj_polarity, obj_polarity});
          }
        }
      }
    }
    return AnalysisGraph(G);
  }

  auto add_node() { return boost::add_vertex(graph); }
  auto add_edge(int i, int j) { boost::add_edge(i, j, graph); }
  auto edges() { return boost::make_iterator_range(boost::edges(graph)); }

  auto predecessors(int i) {
    return boost::make_iterator_range(boost::inv_adjacent_vertices(i, graph));
  }

  auto out_edges(int i) {
    return boost::make_iterator_range(boost::out_edges(i, graph));
  }

  vector<std::pair<int, int>> simple_paths(int i, int j) {
    vector<std::pair<int, int>> paths = {};
    for (auto s : successors(i)) {
      paths.push_back(std::make_pair(i, s));
      for (auto e : simple_paths(s, j)) {
        paths.push_back(e);
      }
    }
    return paths;
  }
  void construct_beta_pdfs() {
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
        KDE(marginalized_responses).resample(default_n_samples);

    for (auto e : edges()) {
      vector<double> all_thetas = {};
      for (auto causalFragment : graph[e].causalFragments) {
        auto subj_adjective = causalFragment.subj_adjective;
        auto obj_adjective = causalFragment.obj_adjective;
        auto subj_responses =
            lmap([&](auto x) { return x * causalFragment.subj_polarity; },
                 get(adjective_response_map,
                     subj_adjective,
                     marginalized_responses));
        auto obj_responses = lmap(
            [&](auto x) { return x * causalFragment.obj_polarity; },
            get(adjective_response_map, obj_adjective, marginalized_responses));
        for (auto [x, y] : iter::product(subj_responses, obj_responses)) {
          all_thetas.push_back(atan2(sigma_Y * y, sigma_X * x));
        }
      }
      graph[e].kde = KDE(all_thetas);
    }
  }

  /*
   * Find all the simple paths between all the paris of nodes of the graph
   */
  void all_paths() {
    auto verts = vertices();

    for_each(verts, [&](int start) {
      for_each(verts, [&](int end) {
        if (start != end) {
          all_paths_between(start, end);
        }
      });
    });
  }

  /*
   * Prints the simple paths found between all pairs of nodes of the graph
   * Groupd according to the starting and ending vertex.
   * all_paths() should be called before this to populate the paths
   */
  void print_all_paths() {
    auto verts = vertices();

    cout << "All the simple paths of:" << endl;

    for_each(verts, [&](int v) {
      cout << endl << "Paths ending at verex: " << v << endl;

      for (auto influencer : graph[v].influenced_by) {
        for (auto path : influencer.second) {
          for (auto edge : path) {
            cout << "(" << edge.first << ", " << edge.second << ") ";
          }
          cout << endl;
        }
        cout << "----" << endl;
      }
    });
  }

  auto sample_from_prior() {
    vector<std::pair<int, int>> node_pairs;

    // Get all length-2 permutations of nodes in the graph
    for (auto [i, j] : iter::product(vertices(), vertices())) {
      if (i != j) {
        node_pairs.push_back(std::make_pair(i, j));
      }
    }

    unordered_map<int, unordered_map<int, vector<std::pair<int, int>>>>
        simple_path_dict;
    for (auto [i, j] : node_pairs) {
      int cutoff = 4;
      int depth = 0;
      vector<std::pair<int, int>> paths;
    }
  }

  auto print_nodes() {
    for_each(vertices(), [&](auto v) { cout << v << endl; });
  }
  auto print_edges() {
    for_each(edges(), [&](auto e) {
      cout << "(" << source(e, graph) << ", " << target(e, graph) << ")"
           << endl;
    });
  }

  auto to_dot() {
    write_graphviz(
        cout, graph, make_label_writer(boost::get(&Node::name, graph)));
  }
};
