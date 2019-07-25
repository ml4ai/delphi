#pragma once

#include <optional>
#include <exception>
#include "kde.hpp"
#include "kde.hpp"
#include "random_variables.hpp"

template <class T> void print(T x) { std::cout << x << std::endl; }

template <class T> void printVec(std::vector<T> xs) {
  for (auto x : xs) {
    print(x);
  }
}


struct IndicatorNotFoundException : public std::exception 
{
  std::string msg;

  IndicatorNotFoundException( std::string msg ) : msg(msg) { }

  const char * what() const throw () {
    return this->msg.c_str();
  }
};


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

  void add_indicator( string indicator, string source )
  {
    //TODO: What if this indicator already exists?
    //      At the moment only the last indicator is recorded
    //      in the indicator_names map
    //What if this indicator already exists?
    //*Loren: We just say it's already attached and do nothing.
    // As of right now, we are only attaching one indicator per node but even
    // if we were attaching multiple indicators to one node, I can't yet think
    // of a case where the numerical id (i.e. the order) matters. If we do come
    // across that case, we will just write a function that swaps ids.*
    if( indicator_names.find( indicator ) != indicator_names.end() )
    {
      std::cout << indicator << " already attached to " << name << std::endl;
      return;
    }

    indicator_names[ indicator ] = indicators.size();
    indicators.push_back( Indicator( indicator, source ));
  }


  Indicator get_indicator( string indicator )
  {
    try 
    {
      return indicators[ indicator_names.at( indicator )];
    }
    catch (const std::out_of_range &oor) 
    {
      throw IndicatorNotFoundException( indicator );   
    }
  }


  void replace_indicator( string indicator_old, string indicator_new, string source )
  {
    auto map_entry =  indicator_names.extract( indicator_old );

    if( map_entry ) // indicator_old is in the map
    {
      // Update the map entry and add the new indicator
      // in place of the old indicator
      map_entry.key() = indicator_new;
      indicator_names.insert( move( map_entry ));
      indicators[ map_entry.mapped() ] = Indicator( indicator_new, source );
    }
    else // indicator_old is not attached to this node
    {
      std::cout << "Node::replace_indicator - indicator " << indicator_old << " is not attached to node " << name << std::endl;
      std::cout << "\tAdding indicator " << indicator_new << " afresh\n";
      add_indicator( indicator_new, source );
    }
  }
  

  void set_indicator_source( string indicator, string source )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::set_indicator_source()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tsource: " << source << " cannot be set" << std::endl;
    }
    indicators[ind_index].set_source(source);
  }


  string get_indicator_source( string indicator )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::get_indicator_source()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tsource could not be retrieved" << std::endl;
    }
    return indicators[ind_index].get_source();
  }


  void set_indicator_unit( string indicator, string unit )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::set_indicator_unit()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tunit: " << unit << " cannot be set" << std::endl;
    }
    indicators[ind_index].set_unit(unit);
  }


  string get_indicator_unit( string indicator )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::get_indicator_unit()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tunit could not be retrieved" << std::endl;
    }
    return indicators[ind_index].get_unit();
  }


  void set_indicator_mean( string indicator, double mean )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::set_indicator_mean()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tmean: " << mean << " cannot be set" << std::endl;
    }
    indicators[ind_index].set_mean(mean);
  }


  double get_indicator_mean( string indicator )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::get_indicator_mean()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tmean could not be retrieved" << std::endl;
    }
    return indicators[ind_index].get_mean();
  }


  void set_indicator_value( string indicator, double value )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::set_indicator_value()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tvalue: " << value << " cannot be set" << std::endl;
    }
    indicators[ind_index].set_value(value);
  }


  double get_indicator_value( string indicator )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::get_indicator_value()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tvalue could not be retrieved" << std::endl;
    }
    return indicators[ind_index].get_value();
  }


  void set_indicator_stdev( string indicator, double stdev )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::set_indicator_stdev()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tstdev: " << stdev << " cannot be set" << std::endl;
    }
    indicators[ind_index].set_stdev(stdev);
  }


  double get_indicator_stdev( string indicator )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::get_indicator_stdev()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tstdev could not be retrieved" << std::endl;
    }
    return indicators[ind_index].get_stdev();
  }

  //uses temporary time type
  void set_indicator_time( string indicator, string time )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::set_indicator_time()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\ttime: " << time << " cannot be set" << std::endl;
    }
    indicators[ind_index].set_time(time);
  }

  //uses temporary time type
  string get_indicator_time( string indicator )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::get_indicator_time()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\ttime could not be retrieved" << std::endl;
    }
    return indicators[ind_index].get_time();
  }


  void set_indicator_aggaxes( string indicator, vector< std::string > aggaxes )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::set_indicator_aggaxes()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\taggaxes: given aggaxes cannot be set" << std::endl;
    }
    indicators[ind_index].set_aggaxes(aggaxes);
  }


  vector< std::string > get_indicator_aggaxes( string indicator )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::get_indicator_aggaxes()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\taggaxes could not be retrieved" << std::endl;
    }
    return indicators[ind_index].get_aggaxes();
  }


  void set_indicator_aggregation_method( string indicator, string aggregation_method )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::set_indicator_aggregation_method()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\taggregation_method: " << aggregation_method << " cannot be set" << std::endl;
    }
    indicators[ind_index].set_aggregation_method(aggregation_method);
  }


  string get_indicator_aggregation_method( string indicator )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::get_indicator_aggregation_method()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\taggregation_method could not be retrieved" << std::endl;
    }
    return indicators[ind_index].get_aggregation_method();
  }


  void set_indicator_timeseries( string indicator, double timeseries )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::set_indicator_timeseries()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\ttimeseries: " << timeseries << " cannot be set" << std::endl;
    }
    indicators[ind_index].set_timeseries(timeseries);
  }


  double get_indicator_timeseries( string indicator )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::get_indicator_timeseries()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\ttimeseries could not be retrieved" << std::endl;
    }
    return indicators[ind_index].get_timeseries();
  }


  void set_indicator_samples( string indicator, vector< double > samples )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::set_indicator_samples()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tsamples: given samples cannot be set" << std::endl;
    }
    indicators[ind_index].set_samples(samples);
  }


  vector< double > get_indicator_samples( string indicator )
  {
    int ind_index;
    try {
      ind_index = indicator_names[indicator];
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Node::get_indicator_samples()\n"
                << "\tIndicator: " << indicator << " does not exist\n";
      std::cerr << "\tsamples could not be retrieved" << std::endl;
    }
    return indicators[ind_index].get_samples();
  }



  
  //Utility function that clears the indicators vector and the name map.
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
