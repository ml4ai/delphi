#ifndef RANDOM_VARIABLES_HPP
#define RANDOM_VARIABLES_HPP

#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>

using std::string, std::vector;


class RV 
{
  public:
    string name;
    double value;
    vector< double > dataset;


    RV( string name ) : name( name ) { }


    double sample()
    {
      vector< double > s;

      std::sample( this->dataset.begin(), this->dataset.end(), std::back_inserter( s ), 1, std::mt19937{ std::random_device{}() });

      return s[0];
    }

};


class LatentVar : public RV 
{
  public:
    double partial_t;

    LatentVar() : RV( "" ) { }
    LatentVar( string name ) : RV( name ) { }
};


/**
 * The Indicator class represents an abstraction of a concrete, tangible
 * quantity that is in some way representative of a higher level concept (i.e.
 * a node in an :class:`delphi.AnalysisGraph.AnalysisGraph` object.)
 *
 * @param source: The source database (FAO, WDI, etc.)
 * @param unit: The units of the indicator.
 * @param mean: The mean value of the indicator (for performing conditional
 *             forecasting queries on the model.)
 * @param value: The current value of the indicator (used while performing inference)
 * @param stdev: The standard deviation of the indicator.
 * @param time: The time corresponding to the parameterization of the indicator.
 * @param aggaxes: A list of axes across which the indicator values have been
 *                aggregated. Examples: 'month', 'year', 'state', etc.
 * @param aggregation_method: The method of aggregation across the aggregation axes.
 *                           Currently defaults to 'mean'.
 * @param timeseries: A time series for the indicator.
 */
class Indicator : public RV 
{
  public:
      string                 source = "";
      string                 unit = "";
      double                 mean = 0;
      double                 value = 0;
      double                 stdev = 0;
      string                 time = ""; // TODO: Need a proper type. There is one in c++20
      vector< std:: string > aggaxes = {};
      string                 aggregation_method = "mean";
      double                 timeseries = 0;
      vector< double >       samples = {};

  Indicator() : RV( "" ) { }

  Indicator(
      string                 name,
      string                 source = "",
      string                 unit = "",
      double                 mean = 0,
      double                 value = 0,
      double                 stdev = 0,
      string                 time = "", // TODO: Need a proper type. There is one in c++20
      vector< std:: string > aggaxes = {},
      string                 aggregation_method = "mean",
      double                 timeseries = 0,
      vector< double >       samples = {}
      ) :
          RV( name ),
          source( source ),
          unit( unit ),
          mean( mean ),
          value( value ),
          stdev( stdev ),
          time( time ), // TODO: Need a proper type. There is one in c++20
          aggaxes( aggaxes ),
          aggregation_method( aggregation_method ),
          timeseries( timeseries ),
          samples( samples )
  { }

  string get_name( )
  {
    return this->name;
  }


  void set_source( string source )
  {
    this->source = source;
  }


  string get_source( )
  {
    return this->source;
  }


  void set_unit( string unit )
  {
    this->unit = unit;
  }


  string get_unit( )
  {
    return this->unit;
  }


  void set_mean( double mean )
  {
    this->mean = mean;
  }


  double get_mean( )
  {
    return this->mean;
  }


  void set_value( double value )
  {
    this->value = value;
  }


  double get_value( )
  {
    return this->value;
  }


  void set_stdev( double stdev )
  {
    this->stdev = stdev;
  }


  double get_stdev( )
  {
    return this->stdev;
  }

  //uses temporary time type
  void set_time( string time )
  {
    this->time = time;
  }

  //uses temporary time type
  string get_time( )
  {
    return this->time;
  }


  void set_aggaxes( vector< std::string > aggaxes )
  {
    this->aggaxes = aggaxes;
  }


  vector< std::string > get_aggaxes( )
  {
    return this->aggaxes;
  }


  void set_aggregation_method( string aggregation_method )
  {
    this->aggregation_method = aggregation_method;
  }


  string get_aggregation_method( )
  {
    return this->aggregation_method;
  }


  void set_timeseries( double timeseries )
  {
    this->timeseries = timeseries;
  }


  double get_timeseries( )
  {
    return this->timeseries;
  }


  void set_samples( vector< double > samples )
  {
    this->samples = samples;
  }


  vector< double > get_samples( )
  {
    return this->samples;
  }


  void set_default_unit()
  {
    sqlite3 *db;
    int rc = sqlite3_open(std::getenv("DELPHI_DB"), &db);
    if (rc){
      fmt::print("Could not open db");
      return;
    }
    std::vector< std::string > units;
    sqlite3_stmt *stmt;
    string query =
      "select Unit from indicator where `Variable` like '" + name + "'";
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      string ind_unit = std::string(
          reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
      units.push_back(ind_unit);
    }
    if (!units.empty()) {
      std::sort(units.begin(),units.end());
      this->unit = units.front();
    }
    else {
      std::cerr << "Error: Indicator::set_indicator_default_unit()\n"
                << "\tIndicator: No data exists for " << name << std::endl;
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
  }

};
#endif
