#pragma once


using namespace std;

// look at log likelihoods and see if they vary within a given delta.
class TrainingStopper {


  private:
    size_t n_samples = 0;
    double delta_magnitude = 0.0;

  public:
    TrainingStopper();
    ~TrainingStopper(){}
    bool stop_training(std::vector<double> log_likelihoods, int sample_idx);

};
