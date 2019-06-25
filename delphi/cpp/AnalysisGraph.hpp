#ifndef ANALYSISGRAPH_H
#define ANALYSISGRAPH_H

#include <nlohmann/json.hpp>
#include <kde.hpp>
#include <optional>

/** \def COUT(x)
    \brief A macro that prints \a x to standard output (stdout).
*/
#define COUT(x) cout << x << endl;
/** \def CERR(x)
    \brief A macro that prints \a x to standard error (stderr).
*/
#define CERR(x) cerr << x << endl;

using std::pair 
    , std::string
    , std::optional
    , std::vector
    , std::unordered_map
    , boost::adjacency_list
    , boost::vecS
    , boost::setS
    , boost::listS
    , boost::directedS
;


struct CausalFragment {
  string subj_adjective;
  string obj_adjective;
  // Here we assume that unknown polarities are set to 1.
  int subj_polarity = 1;
  int obj_polarity = 1;
};


struct Edge {
  string name;
  optional<KDE> kde;
  vector<CausalFragment> causalFragments= {};
};

struct Node {
  string name;
};

struct GraphData {
  string name;
};


typedef adjacency_list<setS, vecS, directedS, Node, Edge, GraphData> DiGraph;

#endif
