#include "AnalysisGraph.hpp"

using namespace std;

//1069
void AnalysisGraph::partition_data_according_to_period(int hn_id,
                                                       std::vector<double> &mean_sequence,
                                                       std::vector<int> &ts_sequence) {
    Node &hn = (*this)[hn_id];
    Indicator &hn_ind = hn.indicators[0];
    vector<double> scaled_obs;

    for (int ts: ts_sequence) {
        vector<double> &obs_at_ts = this->observed_state_sequence[ts][hn_id][0];

        scaled_obs.clear();
        transform(obs_at_ts.begin(), obs_at_ts.end(),
                  scaled_obs.begin(),
                  [&](double v){return v / hn_ind.mean;});

        // For all the concepts ts = 0 is the start
        int partition = ts % hn.period;
        hn.partitioned_data[partition].first.push_back(ts);
        hn.partitioned_data[partition].second.insert(hn.partitioned_data[partition].second.end(),
                                                     scaled_obs.begin(),
                                                     scaled_obs.end());
    }
}