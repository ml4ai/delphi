#include "AnalysisGraph.hpp"
#include "spdlog/spdlog.h"

using namespace std;
using namespace delphi::utils;
using namespace fmt::literals;
using spdlog::debug;
using spdlog::error;


/*
 ============================================================================
 Public: Graph Building
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
        // Note: This is an overloaded private method that takes in a vertex id
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

