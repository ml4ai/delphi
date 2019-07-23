#pragma once

#include "kde.hpp"
#include <optional>
#include "kde.hpp"
#include "random_variables.hpp"

template <class T> void print(T x) { std::cout << x << std::endl; }

template <class T> void printVec(std::vector<T> xs) {
  for (auto x : xs) {
    print(x);
  }
}


struct Event {
  std::string adjective;
  int polarity;
  std::string concept_name;

  Event( std::string adj, int pol, std::string con_name )
    : adjective{ adj }, polarity{ pol }, concept_name{ con_name }
  {
  }

  Event( std::tuple< string, int, string > evnt )
  {
    adjective = std::get< 0 >( evnt );
    polarity = std::get< 1 >( evnt );
    concept_name = std::get< 2 >( evnt );
  }
};


struct Statement {
  Event subject;
  Event object;
};


/*
struct CausalFragment {
  std::string subj_adjective;
  std::string obj_adjective;
  // Here we assume that unknown polarities are set to 1.
  int subj_polarity{1};
  int obj_polarity{1};
};
*/


struct Edge {
  std::string name;
  // TODO: Why kde is optional?
  // According to AnalysisGraph::construct_beta_pdfs()
  // it seems all the edges have a kde
  std::optional<KDE> kde;
  //std::vector<CausalFragment> causalFragments = {};

  std::vector< Statement > evidence;

  // The current β for this edge
  // TODO: Need to decide how to initialize this or
  // decide whethr this is the correct way to do this.
  double beta = 1.0;
};

struct Node {
  std::string name = "";
  bool visited;
  LatentVar rv;

  std::vector< Indicator > indicators;
  // Maps each indicator name to its index in the indicators vector
  std::map< std::string, int > indicator_names;

  void add_indicator( string indicator, string source, bool replace = true, int replace_index = 0 )
  {
    //TODO: What if this indicator already exists?
    //      At the moment only the last indicator is recorded
    //      in the indicator_names map
    if( indicator_names.find( indicator ) != indicator_names.end() )
    {
      std::cout << indicator << " already attached to " << name << std::endl;
      return;
    }
    if (replace)
    {
      if ( replace_index >= indicators.size())
      {
        std::cout << "Replace index is out of bounds, adding " << indicator << " to " << name << " instead" << std::endl;
        indicator_names [ indicator ] = indicators.size();
        indicators.push_back( Indicator( indicator, source ));
        return;
      }
      string to_be_replaced;
      for (auto [ name, idx ] : indicator_names)
      {
        if (indicator_names[name] == replace_index)
        {
          to_be_replaced = name;
        }
      }
      indicator_names.erase(to_be_replaced);
      indicator_names[indicator] = replace_index;
      indicators[replace_index] = Indicator( indicator, source );
      return;
    }
    indicator_names[ indicator ] = indicators.size();
    indicators.push_back( Indicator( indicator, source ));
  }

  void clear_indicators()
  {
    indicators.clear();
    indicator_names.clear();
  }
};

struct GraphData {
  std::string name;
};

typedef boost::adjacency_list<boost::setS,
                              boost::vecS,
                              boost::bidirectionalS,
                              Node,
                              Edge,
                              GraphData>
    DiGraph;
