#pragma once

#include <gvc.h>
#include <string>

/** Syntactic sugar for setting Agraph_t component properties */
namespace delphi::gv {


void set_property(Agnode_t *node,
                  std::string property_name,
                  std::string property_value);

void set_property(Agedge_t *edge,
                  std::string property_name,
                  std::string property_value);

void set_property(Agraph_t *g,
                  int kind,
                  std::string property_name,
                  std::string property_value);

Agnode_t *add_node(Agraph_t *g, std::string node_name);
} // namespace delphi::gv
