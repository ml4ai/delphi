#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <vector>
#include <fmt/format.h>

namespace utils {

using std::cout, std::endl, std::vector, std::string, std::ifstream,
    std::unordered_map;

template <class T> void printVec(vector<T> xs) {
  for (auto x : xs) {
    fmt::print(x);
  }
}
template <class Key, class Value> bool hasKey(unordered_map<Key, Value> umap, Key key) {
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
} // namespace utils
