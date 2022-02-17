#include "AnalysisGraph.hpp"

using namespace std;

//1084
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

// 1087
pair<Eigen::MatrixXd, Eigen::VectorXd>
    assemble_sinusoidal_generating_LDS(unsigned short components) {
    unsigned short comps_2 = components * 2;
    Eigen::MatrixXd A_sin_k = Eigen::MatrixXd::Zero(comps_2, comps_2);
    Eigen::VectorXd s0_sin_k = Eigen::VectorXd::Zero(comps_2);

    for (int i = 0; i < components; i++) {
        int i2 = i * 2;
        int i2p1 = i2 + 1;
        int ip1 = i + 1;
        A_sin_k(i2, i2p1) = 1;
        A_sin_k(i2p1, i2) = -(ip1 * ip1);
        s0_sin_k(i2p1) = ip1 * cos(ip1 * M_PI); //#(-1)**(ip1)*ip1
    }

    return make_pair(A_sin_k, s0_sin_k);
}
