#pragma once
#include <random>

class RNG{
public:
  static RNG* rng();
  void set_seed(int seed);
  int get_seed();
  std::mt19937 get_RNG();

private:
  RNG(){
    std::random_device rd;
    this->random_seed = rd();
    this->gen = std::mt19937(this->random_seed);
  };
  RNG(RNG const&);
  RNG& operator=(RNG const&);
  static RNG* m_pInstance;
  int random_seed;
  std::mt19937 gen;

};

RNG* RNG::m_pInstance = NULL;

RNG* RNG::rng()
{
 if(!m_pInstance)
   m_pInstance = new RNG;

 return m_pInstance;
}

void RNG::set_seed(int seed)
{
  this->random_seed = seed;
  this->gen = std::mt19937(this->random_seed);
}

int RNG::get_seed()
{
  return this->random_seed;
}

std::mt19937 RNG::get_RNG()
{
  return this->gen;
}
