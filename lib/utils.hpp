#pragma once

#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>
#include <boost/range/numeric.hpp>

namespace utils {

using std::cout, std::endl, std::vector, std::string, std::ifstream,
    std::unordered_map;

template <class T> void printVec(vector<T> xs) {
  for (T x : xs) {
    cout << x << ", " << endl;
  }
}

template <class Key, class Value>
bool hasKey(unordered_map<Key, Value> umap, Key key) {
  return umap.count(key) != 0;
}

template <class Key, class Value>
Value get(unordered_map<Key, Value> umap, Key key, Value default_value) {
  return hasKey(umap, key) ? umap[key] : default_value;
}

template <class V, class Iterable> vector<V> list(Iterable xs) {
  vector<V> xs_copy;
  for (auto x : xs) {
    xs_copy.push_back(x);
  }
  return xs_copy;
}

template <class F, class V> vector<V> lmap(F f, vector<V> vec) {
  vector<V> transformed_vector;
  for (V x : vec) {
    transformed_vector.push_back(f(x));
  }
  return transformed_vector;
}

nlohmann::json load_json(string filename) {
  ifstream i(filename);
  nlohmann::json j;
  i >> j;
  return j;
}

/**
 * Returns the square of a number.
 */
double sqr(double x) { return x * x; }

/**
 * Returns the sum of a vector of doubles.
 */
double sum(std::vector<double> v) { return boost::accumulate(v, 0.0); }


/**
 * Returns the arithmetic mean of a vector of doubles.
 */
double mean(std::vector<double> v) { return sum(v) / v.size(); }

} // namespace utils
