#ifndef ANALYSISGRAPH_H
#define ANALYSISGRAPH_H

#include <nlohmann/json.hpp>
#include <kde.hpp>

/** \def COUT(x)
    \brief A macro that prints \a x to standard output (stdout).
*/
#define COUT(x) cout << x << endl;
/** \def CERR(x)
    \brief A macro that prints \a x to standard error (stderr).
*/
#define CERR(x) cerr << x << endl;

namespace delphi {
  using std::cout,
        std::endl,
        std::unordered_map,
        std::pair,
        std::string,
        std::ifstream,
        std::stringstream,
        std::vector,
        std::map,
        boost::inner_product,
        boost::adjacency_list,
        boost::edge,
        boost::add_edge,
        boost::vecS,
        boost::directedS,
        boost::edges,
        boost::source,
        boost::target,
        boost::get,
        boost::make_label_writer,
        boost::write_graphviz,
        boost::range::for_each,
        boost::lambda::make_const
  ;

  using json = nlohmann::json;
}

using namespace delphi;

struct Node {
  string name;
};

struct CAGEdge {
  string name;
  KDE kde;
};

struct Model {
  string name;
};

typedef pair<int, int> Edge;
typedef adjacency_list<vecS, vecS, directedS, Node, CAGEdge, Model> DiGraph;

#endif
