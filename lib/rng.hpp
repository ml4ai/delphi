#pragma once

#include <random>

class RNG {
public:
  static RNG *rng();
  void set_seed(int seed);
  int get_seed();
  std::mt19937 get_RNG();
  static void release_instance();

private:
  RNG();
  ~RNG(){};
  RNG(RNG const &);
  RNG &operator=(RNG const &);
  static RNG *m_pInstance;
  int random_seed;
  std::mt19937 gen;
  static void add_ref();
  static void release_ref();
  static int counter;
};
