#pragma once

#include <random>

class RNG {
public:
  static RNG *rng();
  void set_seed(int seed);
  int get_seed();
  std::mt19937 get_RNG();

private:
  RNG();
  RNG(RNG const &);
  RNG &operator=(RNG const &);
  static RNG *m_pInstance;
  int random_seed;
  std::mt19937 gen;
};
