#include "rng.hpp"

RNG::RNG() {
    std::random_device rd;
    this->random_seed = rd();
    this->gen = std::mt19937(this->random_seed);
}

RNG *RNG::m_pInstance = NULL;
int RNG::counter = 0;

RNG *RNG::rng() {
  if (!m_pInstance)
    m_pInstance = new RNG;

  return m_pInstance;
}

void RNG::set_seed(int seed) {
  this->random_seed = seed;
  this->gen = std::mt19937(this->random_seed);
}

int RNG::get_seed() { return this->random_seed; }

std::mt19937 RNG::get_RNG() { 
  // Track users
  add_ref();

  return this->gen;
}

void RNG::add_ref() {
  ++counter;
}

void RNG::release_ref() {
  --counter;
}

void RNG::release_instance() {
  release_ref();

  if (counter == 0 && m_pInstance != NULL) {
    delete m_pInstance;
    m_pInstance = NULL;
  }
}
