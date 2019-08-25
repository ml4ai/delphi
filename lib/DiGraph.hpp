#pragma once

#include <string>
#include "random_variables.hpp"
#include "kde.hpp"
#include <boost/graph/adjacency_list.hpp>

class IndicatorNotFoundException : public std::exception {
  public:
  std::string msg;

  IndicatorNotFoundException(std::string msg) : msg(msg) {}

  const char *what() const throw() { return this->msg.c_str(); }
};

class Node {
  public:
  std::string name = "";
  bool visited;
  LatentVar rv;

  std::vector<Indicator> indicators;
  // Maps each indicator name to its index in the indicators vector
  std::map<std::string, int> indicator_names;

  void add_indicator(std::string indicator, std::string source) {
    // TODO: What if this indicator already exists?
    //      At the moment only the last indicator is recorded
    //      in the indicator_names map
    // What if this indicator already exists?
    //*Loren: We just say it's already attached and do nothing.
    // As of right now, we are only attaching one indicator per node but even
    // if we were attaching multiple indicators to one node, I can't yet think
    // of a case where the numerical id (i.e. the order) matters. If we do come
    // across that case, we will just write a function that swaps ids.*
    if (indicator_names.find(indicator) != indicator_names.end()) {
      std::cout << indicator << " already attached to " << name << std::endl;
      return;
    }

    indicator_names[indicator] = indicators.size();
    indicators.push_back(Indicator(indicator, source));
  }


  void delete_indicator(std::string indicator) {
    if (indicator_names.find(indicator) != indicator_names.end()) {
      int ind_index = indicator_names[indicator];
      indicator_names.clear();
      indicators.erase(indicators.begin() + ind_index);
      //The values of the map object have to align with the vecter indexes.
      for (int i = 0; i < indicators.size(); i++) {
        indicator_names[indicators[i].get_name()] = i;
      }
    }
    else {
      std::cout << "There is no indicator  " << indicator << "attached to " << name << std::endl;  
    }
  }


  Indicator get_indicator(std::string indicator) {
    try {
      return indicators[indicator_names.at(indicator)];
    }
    catch (const std::out_of_range &oor) {
      throw IndicatorNotFoundException(indicator);
    }
  }
  

  void replace_indicator(std::string indicator_old,
                         std::string indicator_new,
                         std::string source) {

    auto map_entry = indicator_names.extract(indicator_old);

    if (map_entry) // indicator_old is in the map
    {
      // Update the map entry and add the new indicator
      // in place of the old indicator
      map_entry.key() = indicator_new;
      indicator_names.insert(move(map_entry));
      indicators[map_entry.mapped()] = Indicator(indicator_new, source);
    }
    else // indicator_old is not attached to this node
    {
      std::cout << "Node::replace_indicator - indicator " << indicator_old
                << " is not attached to node " << name << std::endl;
      std::cout << "\tAdding indicator " << indicator_new << " afresh\n";
      add_indicator(indicator_new, source);
    }
  }

  // Utility function that clears the indicators vector and the name map.
  void clear_indicators() {
    indicators.clear();
    indicator_names.clear();
  }
};

class Event {
  public:
  std::string adjective;
  int polarity;
  std::string concept_name;

  Event(std::string adj, int pol, std::string con_name)
      : adjective{adj}, polarity{pol}, concept_name{con_name} {}

  Event(std::tuple<std::string, int, std::string> evnt) {
    adjective = std::get<0>(evnt);
    polarity = std::get<1>(evnt);
    concept_name = std::get<2>(evnt);
  }
};

class Statement {
  public:
  Event subject;
  Event object;

  Statement() :
    subject(Event("", 0, "" )),
    object(Event("", 0, ""))
    {}

  Statement(Event sub, Event obj) :
    subject(sub),
    object(obj)
    {}
};

class Edge {
  public:
  std::string name;
  // TODO: Why kde is optional?
  // According to AnalysisGraph::construct_beta_pdfs()
  // it seems all the edges have a kde
  std::optional<KDE> kde;
  // std::vector<CausalFragment> causalFragments = {};

  std::vector<Statement> evidence;

  // The current β for this edge
  // TODO: Need to decide how to initialize this or
  // decide whethr this is the correct way to do this.
  double beta = 1.0;
  void change_polarity(int subject_polarity, int object_polarity) {
    for (Statement stmt : evidence) {
      stmt.subject.polarity = subject_polarity;
      stmt.object.polarity = object_polarity;
    }
  }
};

class GraphData {
  public:
  std::string name;
};

typedef boost::adjacency_list<boost::setS,
                              boost::vecS,
                              boost::bidirectionalS,
                              Node,
                              Edge,
                              GraphData>
    DiGraph;

typedef boost::iterator_range<boost::adjacency_iterator<DiGraph, long unsigned int, boost::detail::out_edge_iter<std::_Rb_tree_const_iterator<boost::detail::stored_edge_iter<long unsigned int, std::_List_iterator<boost::list_edge<long unsigned int, Edge> >, Edge> >, long unsigned int, boost::detail::edge_desc_impl<boost::bidirectional_tag, long unsigned int>, long int>, long int> >
    NEIGHBOR_ITERATOR;
