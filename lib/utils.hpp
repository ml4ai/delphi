#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>

namespace utils {
template <class T> void print(T x) { std::cout << x << std::endl; }

template <class T> void printVec(std::vector<T> xs) {
  for (auto x : xs) {
    print(x);
  }
}
template <class T, class U> bool hasKey(std::unordered_map<T, U> umap, T key) {
  return umap.count(key) != 0;
}

template <class T, class U>
U get(std::unordered_map<T, U> umap, T key, U default_value) {
  return hasKey(umap, key) ? umap[key] : default_value;
}

nlohmann::json load_json(std::string filename) {
  std::ifstream i(filename);
  nlohmann::json j;
  i >> j;
  return j;
}
} // namespace utils
