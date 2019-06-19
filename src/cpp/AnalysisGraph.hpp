#ifndef ANALYSISGRAPH_H
#define ANALYSISGRAPH_H

#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/numeric.hpp>
#include <boost/range/adaptors.hpp>

#include <nlohmann/json.hpp>
#define COUT(x) cout << x << endl;
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
        boost::accumulate,
        boost::adaptors::transform,
        boost::adaptors::transformed,
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
        boost::lambda::_1,
        boost::lambda::_2,
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
};

struct Model {
  string name;
};

typedef pair<int, int> Edge;
typedef adjacency_list<vecS, vecS, directedS, Node, CAGEdge, Model> DiGraph;

#endif
