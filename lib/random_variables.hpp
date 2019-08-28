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
 * @parm source: The source database (FAO, WDI, etc.)
 * @parm unit: The units of the indicator.
 * @parm mean: The mean value of the indicator (for performing conditional
 *             forecasting queries on the model.)
 * @parm value: The current value of the indicator (used while performing inference)
 * @parm stdev: The standard deviation of the indicator.
 * @parm time: The time corresponding to the parameterization of the indicator.
 * @parm aggaxes: A list of axes across which the indicator values have been
 *                aggregated. Examples: 'month', 'year', 'state', etc.
 * @parm aggregation_method: The method of aggregation across the aggregation axes.
 *                           Currently defaults to 'mean'.
 * @parm timeseries: A time series for the indicator.
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
};
