#include "AnalysisGraph.hpp"

using namespace std;
using namespace delphi::utils;
using fmt::print;
using namespace fmt::literals;


/*
 ============================================================================
 Public: Graph Building
 ============================================================================
*/

int AnalysisGraph::add_node(string concept) {
  if (!in(this->name_to_vertex, concept)) {
    int v = boost::add_vertex(this->graph);
    this->name_to_vertex[concept] = v;
    (*this)[v].name = concept;
    this->independent_nodes.insert(v);
  }

  return  this->name_to_vertex[concept];
}

bool AnalysisGraph::add_edge(CausalFragment causal_fragment) {
  Event subject = Event(causal_fragment.first);
  Event object = Event(causal_fragment.second);

  string subj_name = subject.concept_name;
  string obj_name = object.concept_name;

  if (subj_name.compare(obj_name) != 0) { // Guard against self loops
    // Add the nodes to the graph if they are not in it already
    int subj_id = this->add_node(subj_name);
    int obj_id = this->add_node(obj_name);

    // Object is a dependent node
    this->dependent_nodes.insert(obj_id);

    // If Object had been an independent node, it no linger is
    if (this->independent_nodes.find(obj_id) != this->independent_nodes.end()) {
      this->independent_nodes.erase(obj_id);
    }

    // If Subject was not a dependent node, it is independent
//    if (this->dependent_nodes.find(subj_id) == this->dependent_nodes.end()) {
//      this->independent_nodes.insert(subj_id);
//    }

    auto [e, exists] = this->add_edge(subj_name, obj_name);
    this->graph[e].evidence.push_back(Statement{subject, object});

    return true;
  }
  else {
    print("AnalysisGraph::add_edge\n"
          "\tWARNING: Prevented adding a self loop for the concept {}\n",
          subj_name);
    return false;
  }
}

void AnalysisGraph::add_edge(CausalFragmentCollection causal_fragments) {
  EventCollection subjects = causal_fragments.first;
  EventCollection objects = causal_fragments.second;
  string subj_name = get<2>(subjects);
  string obj_name = get<2>(objects);
  int num_subj = get<0>(subjects).size();
  int num_obj = get<0>(objects).size();

  if (subj_name.compare(obj_name) != 0) { // Guard against self loops
    // Add the nodes to the graph if they are not in it already
    this->add_node(subj_name);
    this->add_node(obj_name);

    auto [e, exists] = this->add_edge(subj_name, obj_name);

    string subject_adj;
    string object_adj ;
    int subject_pol;
    int object_pol;
    for(int stmt = 0; stmt < max(num_subj, num_obj); stmt++){
      if(stmt < num_subj){
        subject_adj = get<0>(subjects)[stmt] ;
        subject_pol = get<1>(subjects)[stmt];
      } else {
        subject_adj = "None";
        subject_pol = 1;
      }
      if(stmt < num_obj){
        object_adj = get<0>(objects)[stmt] ;
        object_pol = get<1>(objects)[stmt];
      } else {
        object_adj = "None";
        object_pol = 1;
      }

      Event subject = Event(subject_adj, subject_pol, subj_name);
      Event object  = Event(object_adj, object_pol, obj_name);

      this->graph[e].evidence.push_back(Statement{subject, object});
    }
  } else {
    print("AnalysisGraph::add_edge\n"
    "\tWARNING: Prevented adding a self loop for the concept {}\n",
    subj_name);
  }
}

pair<EdgeDescriptor, bool> AnalysisGraph::add_edge(int source, int target) {
  // In Boost Graph Library, we are using vecS as the OutEdgeList,
  // So, we have to check whether an edge exists from source to target before
  // adding to prevent adding multiple edges between them in the same direction.
  pair<EdgeDescriptor, bool> edge = boost::edge(source, target, this->graph);

  if (!edge.second) {
    edge = boost::add_edge(source, target, this->graph);

    // Object is a dependent node
    this->dependent_nodes.insert(target);

    // If Object had been an independent node, it no linger is
    if (this->independent_nodes.find(target) != this->independent_nodes.end()) {
      this->independent_nodes.erase(target);
    }
  }

  return edge;
}

pair<EdgeDescriptor, bool> AnalysisGraph::add_edge(int source, string target) {
  return this->add_edge(source, this->get_vertex_id(target));
}

pair<EdgeDescriptor, bool> AnalysisGraph::add_edge(string source, int target) {
  return this->add_edge(this->get_vertex_id(source), target);
}

pair<EdgeDescriptor, bool> AnalysisGraph::add_edge(string source,
                                                   string target) {
  return this->add_edge(this->get_vertex_id(source),
                         this->get_vertex_id(target));
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
    print("AnalysisGraph::remove_nodes()\n"
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

                     if (!in(this->beta2cell, edge_id)) {
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
    print("AnalysisGraph::remove_edges");

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

