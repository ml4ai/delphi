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
    std::chrono::time_point<std::chrono::steady_clock> start_timepoint;
    std::pair<std::vector<std::string>, std::vector<long>> &store;
    struct timespec start_cpu;

  public:
    Timer(std::string task, std::pair<std::vector<std::string>,
          std::vector<long>> &store): store(store), task(task){
      start_timepoint = std::chrono::steady_clock::now();
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_cpu);
    };

    ~Timer() {
        this->stop();
    };

    void stop() {
      std::chrono::time_point<std::chrono::steady_clock> end_timepoint
          = std::chrono::steady_clock::now();

      struct timespec end_cpu;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_cpu);

      long seconds = end_cpu.tv_sec - start_cpu.tv_sec;
      long nanoseconds = end_cpu.tv_nsec - start_cpu.tv_nsec;
      double elapsed_cpu = seconds*1e9 + nanoseconds;

      auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>
                   (start_timepoint).time_since_epoch().count();
      auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>
                 (end_timepoint).time_since_epoch().count();

      this->store.first.push_back(task + "(Wall)");
      this->store.second.push_back(end - start);
      this->store.first.push_back(task + "(CPU)");
      this->store.second.push_back(elapsed_cpu);
    };
};

#endif // DELPHI_TIMER_H
