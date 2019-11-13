#include "AnalysisGraph.hpp"

using namespace std;
using namespace delphi::utils;


/*
 ============================================================================
 Public: Graph Modification
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
