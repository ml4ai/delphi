#include <AnalysisGraph.hpp>
#include <Node.hpp>
#include <graphviz_interface.hpp>
#include <range/v3/all.hpp>

using namespace std;

/*
 ============================================================================
 Private: Graph Visualization (in graphviz.cpp)
 ============================================================================
*/

pair<Agraph_t*, GVC_t*> AnalysisGraph::to_agraph(bool simplified_labels,
                                                 int label_depth,
                                                 string node_to_highlight,
                                                 string rankdir,
                                                 bool show_indicators) {

  using delphi::gv::set_property, delphi::gv::add_node;
  using namespace ranges::views;
  using ranges::end, ranges::to;
  using ranges::views::slice, ranges::views::replace, ranges::max,
      ranges::views::transform;

  Agraph_t* G = agopen(const_cast<char*>("G"), Agdirected, NULL);
  GVC_t* gvc;
  gvc = gvContext();

  // Set global properties
  set_property(G, AGNODE, "shape", "rectangle");
  set_property(G, AGNODE, "style", "rounded");
  set_property(G, AGNODE, "color", "maroon");
  set_property(G, AGRAPH, "dpi", "150");
  set_property(G, AGRAPH, "overlap", "scale");
  set_property(G, AGRAPH, "splines", "true");
  set_property(G, AGRAPH, "rankdir", rankdir);

#if defined __APPLE__
  set_property(G, AGNODE, "fontname", "Gill Sans");
#else
  set_property(G, AGNODE, "fontname", "Helvetica");
#endif

  Agnode_t* src;
  Agnode_t* trgt;
  Agedge_t* edge;

  string source_label;
  string target_label;

  // Add CAG links
  for (auto e : this->edges()) {
    string source_name = this->source(e).name;
    string target_name = this->target(e).name;

    // TODO Implement a refined version of this that checks for set size
    // equality, a la the Python implementation (i.e. check if the length of
    // the nodeset is the same as the length of the set of simplified labels).

    string source_label, target_label;

    if (simplified_labels == true) {
      source_label = source_name | split('/') | slice(end - label_depth, end) |
                     join('/') | replace('_', ' ') | to<string>();
      target_label = target_name | split('/') | slice(end - label_depth, end) |
                     join('/') | replace('_', ' ') | to<string>();
    }
    else {
      source_label = source_name;
      target_label = target_name;
    }

    src = add_node(G, source_name);
    set_property(src, "label", source_label);

    trgt = add_node(G, target_name);
    set_property(trgt, "label", target_label);

    edge = agedge(G, src, trgt, 0, true);
  }

  if (node_to_highlight != "") {
    Agnode_t* node;
    node = add_node(G, node_to_highlight);
    set_property(node, "color", "blue");
  }

  // Add concepts, indicators, and link them.
  if (show_indicators) {
      for (Node& node : this->nodes()) {
          string concept_name = node.name;
          for (auto indicator : node.indicators) {
              src = add_node(G, concept_name);
              trgt = add_node(G, indicator.name);
              set_property(trgt,
                           "label",
                           indicator.name + "\nSource: " + indicator.source +
                               "\nDBN Initialization Value: " +
                               to_string(indicator.mean) +
                               "\nUnit: " + indicator.unit);
              set_property(trgt, "style", "rounded,filled");
              set_property(trgt, "fillcolor", "lightblue");

              edge = agedge(G, src, trgt, 0, true);
          }
      }
  }
  gvLayout(gvc, G, "dot");
  return make_pair(G, gvc);
}

/*
 ============================================================================
 Public: Graph Visualization
 ============================================================================
*/

void AnalysisGraph::to_png(string filename,
                           bool simplified_labels,
                           int label_depth,
                           string node_to_highlight,
                           string rankdir,
                           bool show_indicators) {
  auto [G, gvc] = this->to_agraph(
      simplified_labels, label_depth, node_to_highlight,
        rankdir, show_indicators);
  gvRenderFilename(gvc, G, "png", const_cast<char*>(filename.c_str()));
  gvFreeLayout(gvc, G);
  agclose(G);
  gvFreeContext(gvc);
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
