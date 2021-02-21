#pragma once

#include <string>
#include <vector>
#include "Random_Variables.hpp"

/**
 * The Indicator class represents an abstraction of a concrete, tangible
 * quantity that is in some way representative of a higher level concept (i.e.
 * a node in an :class:`delphi.AnalysisGraph.AnalysisGraph` object.)
 *
 * @param source: The source database (FAO, WDI, etc.)
 * @param unit: The units of the indicator.
 * @param mean: The mean value of the indicator (for performing conditional
 *             forecasting queries on the model.)
 * @param value: The current value of the indicator (used while performing
 * inference)
 * @param stdev: The standard deviation of the indicator.
 * @param time: The time corresponding to the parameterization of the indicator.
 * @param aggaxes: A list of axes across which the indicator values have been
 *                aggregated. Examples: 'month', 'year', 'state', etc.
 * @param aggregation_method: The method of aggregation across the aggregation
 * axes. Currently defaults to 'mean'.
 * @param timeseries: A time series for the indicator.
 */
class Indicator : public RV {
public:
  std::string source = "";
  std::string unit = "";
  double mean = 1;
  double value = 0;
  double stdev = 1;
  std::string time = ""; // TODO: Need a proper type. There is one in c++20
  std::vector<std::string> aggaxes = {};
  std::string aggregation_method = "first";
  double timeseries = 0;
  std::vector<double> samples = {};

  Indicator() : RV("") {}

  Indicator(
      std::string name,
      std::string source = "",
      std::string unit = "",
      double mean = 1,
      double value = 0,
      double stdev = 1,
      std::string time = "", // TODO: Need a proper type. There is one in c++20
      std::vector<std::string> aggaxes = {},
      std::string aggregation_method = "first",
      double timeseries = 0,
      std::vector<double> samples = {})
      : RV(name), source(source), unit(unit), mean(mean), value(value),
        stdev(stdev),
        time(time), // TODO: Need a proper type. There is one in c++20
        aggaxes(aggaxes), aggregation_method(aggregation_method),
        timeseries(timeseries), samples(samples) {}

  std::string get_name() { return this->name; }

  void set_source(std::string source) { this->source = source; }

  std::string get_source() { return this->source; }

  void set_unit(std::string unit) { this->unit = unit; }

  std::string get_unit() { return this->unit; }

  void set_mean(double mean) { this->mean = mean; }

  double get_mean() { return this->mean; }

  void set_value(double value) { this->value = value; }

  double get_value() { return this->value; }

  void set_stdev(double stdev) { this->stdev = stdev; }

  double get_stdev() { return this->stdev; }

  // uses temporary time type
  void set_time(std::string time) { this->time = time; }

  // uses temporary time type
  std::string get_time() { return this->time; }

  void set_aggaxes(std::vector<std::string> aggaxes) {
    this->aggaxes = aggaxes;
  }

  std::vector<std::string> get_aggaxes() { return this->aggaxes; }

  void set_aggregation_method(std::string aggregation_method) {
    this->aggregation_method = aggregation_method;
  }

  std::string get_aggregation_method() { return this->aggregation_method; }

  void set_timeseries(double timeseries) { this->timeseries = timeseries; }

  double get_timeseries() { return this->timeseries; }

  void set_samples(std::vector<double> samples) { this->samples = samples; }

  std::vector<double> get_samples() { return this->samples; }

  void set_default_unit();
};
