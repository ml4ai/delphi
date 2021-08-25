//
// Created by Manujinda Wathugala on 8/25/21.
//

#ifndef DELPHI_TIMER_H
#define DELPHI_TIMER_H

#include <string>
#include <chrono>
#include <vector>

class Timer {
  private:
    std::string task;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_timepoint;
    std::pair<std::vector<std::string>, std::vector<long>> &store;

  public:
    Timer(std::string task, std::pair<std::vector<std::string>,
          std::vector<long>> &store): store(store), task(task){
      start_timepoint = std::chrono::high_resolution_clock::now();
    };

    ~Timer() {
        this->stop();
    };

    void stop() {
      std::chrono::time_point<std::chrono::high_resolution_clock> end_timepoint
          = std::chrono::high_resolution_clock::now();

      auto start = std::chrono::time_point_cast<std::chrono::milliseconds>
                   (start_timepoint).time_since_epoch().count();
      auto end = std::chrono::time_point_cast<std::chrono::milliseconds>
                 (end_timepoint).time_since_epoch().count();

      this->store.first.push_back(task);
      this->store.second.push_back(end - start);
    };
};

#endif // DELPHI_TIMER_H
