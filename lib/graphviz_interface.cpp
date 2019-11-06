#include "graphviz_interface.hpp"
#include <cstdio>

/** Syntactic sugar for setting Agraph_t component properties */
namespace delphi::gv {

void set_property(Agedge_t *edge,
                  std::string property_name,
                  std::string property_value) {
  agsafeset(edge,
            const_cast<char *>(property_name.c_str()),
            const_cast<char *>(property_value.c_str()),
            const_cast<char *>(""));
}

void set_property(Agnode_t *node,
                  std::string property_name,
                  std::string property_value) {
  agsafeset(node,
            const_cast<char *>(property_name.c_str()),
            const_cast<char *>(property_value.c_str()),
            const_cast<char *>(""));
}

void set_property(Agraph_t *g,
                  int kind,
                  std::string property_name,
                  std::string property_value) {
  agattr(g,
         kind,
         const_cast<char *>(property_name.c_str()),
         const_cast<char *>(property_value.c_str()));
}

Agnode_t *add_node(Agraph_t *g, std::string node_name) {
  return agnode(g, const_cast<char *>(node_name.c_str()), 1);
}
} // namespace delphi::gv
