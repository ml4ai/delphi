#include "AnalysisGraph.hpp"
#include "CSVWriter.hpp"

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

/**
   * Generates a vector of all the effective frequencies for a particular period
   * @param components: The number of pure sinusoidal frequencies to generate
   * @param period: The period of the variable(s) being modeled by Fourier
   *                decomposition.
   * @return A vector of all the effective frequencies
 */
vector<double> AnalysisGraph::generate_frequencies_for_period(int components,
                                                              int period) {
    vector<double> freqs(components);

    // λ is the amount we have to stretch/shrink pure sinusoidal curves to make
    // one sine cycle = period radians
    double lambda = 2.0 * M_PI / period;

    // ω is the frequency of each stretched/shrunk sinusoidal curve.
    // With respect to trigonometric function differentiation, the effective
    // frequency of pure sinusoidal curves is λω
    for (int omega = 1; omega <= components; omega++) {
        freqs[omega - 1] = lambda * omega;
    }

    return freqs;
}

/**
   * Assemble the LDS to generate sinusoidals of desired frequencies
   * @param freqs A vector of effective frequencies
   *              (2πω / period; ω = 1, 2, ...)
   * @return pair (base transition matrix, initial state)
   *         0 radians is the initial angle.
 */
pair<Eigen::MatrixXd, Eigen::VectorXd>
AnalysisGraph::assemble_sinusoidal_generating_LDS(const vector<double> &freqs) {
    int comps_2 = freqs.size() * 2;
    Eigen::MatrixXd A_sin = Eigen::MatrixXd::Zero(comps_2, comps_2);
    Eigen::VectorXd s0_sin = Eigen::VectorXd::Zero(comps_2);

    for (int i = 0; i < freqs.size(); i++) {
        int i2 = i * 2;
        int i2p1 = i2 + 1;

        // Assembling a block diagonal matrix
        // with each 2 x 2 block having the format
        // columns ->   2i        2i+1
        //            _________________
        // row 2i    |   0          1 |
        // row 2i+1  | -λω^2        0 |
        // ω = i+1 & λ = 2π/period
        A_sin(i2, i2p1) = 1;
        A_sin(i2p1, i2) = -freqs[i] * freqs[i];

        // Considering t0 = 0 radians, the initial state of the sinusoidal
        // generating LDS is:
        // row 2i    |    sin(λω 0) |  = |  0 |
        // row 2i+1  | λω cos(λω 0) |    | λω |
        // ω = i+1 & λ = 2π/period
        s0_sin(i2p1) = freqs[i];
    }

    return make_pair(A_sin, s0_sin);
}

/*
// 1087
pair<Eigen::MatrixXd, Eigen::VectorXd>
AnalysisGraph::assemble_sinusoidal_generating_LDS(unsigned short components,
                                                  unsigned short period) {
    unsigned short comps_2 = components * 2;
    Eigen::MatrixXd A_sin_k = Eigen::MatrixXd::Zero(comps_2, comps_2);
    Eigen::VectorXd s0_sin_k = Eigen::VectorXd::Zero(comps_2);

    // Assembling a block diagonal matrix
    // with each 2 x 2 block having the format (ω = i + 1)
    // colunms           2i               2i+1
    //            _____________________________
    // row 2i    |        0                 1 |
    // row 2i+1  | -(2πω / period)^2        0 |
    for (int i = 0; i < components; i++) {
        int i2 = i * 2;
        int i2p1 = i2 + 1;
        int ip1 = i + 1;
        double combined_frequency = 2.0 * M_PI * ip1 / period;
        A_sin_k(i2, i2p1) = 1;
        A_sin_k(i2p1, i2) = -combined_frequency * combined_frequency;

        // Considering t0 = 0 radians, the initial state of the sinusoidal
        // generating LDS is:
        // row 2i    |       sin(0)          |  = |       0        |
        // row 2i+1  | (2πω / period) cos(0) |    | (2πω / period) |
        // s0_sin_k(i2p1) = ip1 * cos(ip1 * M_PI); //#(-1)**(ip1)*ip1
        s0_sin_k(i2p1) = combined_frequency;
    }

    return make_pair(A_sin_k, s0_sin_k);
}
 */

/**
   * Generate a matrix of sinusoidal values of all the desired frequencies for
   * all the bin locations.
   * @param A_sin_base: Base transition matrix for sinusoidal generating LDS
   * @param s0_sin: Initial state (0 radians) for sinusoidal generating LDS
   * @param period: Period of the time series being fitted
   * @return A matrix of required sinusoidal values.
   *         row t contains sinusoidals for bin t (radians)
   *         col 2ω contains sin(λω t)
   *         col 2ω+1 contains λω cos(λω t)
   *         with ω = 1, 2, ... & λ = 2π/period
 */
Eigen::MatrixXd
AnalysisGraph::generate_sinusoidal_values_for_bins(const Eigen::MatrixXd &A_sin_base,
                                                   const Eigen::VectorXd &s0_sin,
                                                   int period) {
    // Transition matrix to advance the sinusoidal generating LDS from one bin
    // to the next.
    Eigen::MatrixXd A_sin_step = A_sin_base.exp();

    // A matrix to accumulate sinusoidals of all the frequencies for all the bin
    // locations required for the Fourier curve fitting.
    Eigen::MatrixXd sinusoidals = Eigen::MatrixXd::Zero(s0_sin.size(), period);
    sinusoidals.col(0) = s0_sin;

    // Evolve the sinusoidal generating LDS one step at a time for a whole
    // period to generate all the sinusoidal values required for the Fourier
    // reconstruction of the seasonal time series. Column j provides
    // sinusoidal values required for bin j.
    for (int col = 1; col < period; col++) {
        sinusoidals.col(col) = A_sin_step * sinusoidals.col(col - 1);
    }
    // Transpose the sinusoidal matrix so that row i contains the sinusoidal
    // values for bin i.
    sinusoidals.transposeInPlace();

    return sinusoidals;
}

/**
   * Computes the Fourier coefficients to fit a seasonal curve to partitioned
   * observations using the least square optimization.
   * @param sinusoidals Sinusoidal values of required frequencies at each bin
   *        position. Row b contains all the sinusoidal values for bin b.
   *        sinusoidals(b, 2(i-1))     =    sin(λi b)
   *        sinusoidals(b, 2(i-1) + 1) = λi cos(λi b)
   * @return The Fourier coefficients in the order: α₀, β₁, α₁, β₂, α₂, ...
   *         α₀ is the coefficient for    cos(0)/2  term
   *         αᵢ is the coefficient for λi cos(λi b) term
   *         βᵢ is the coefficient for    sin(λi b) term
   *
   *         with i = 1, 2, ... & λ = 2π/period & b = 0, 1, ..., period - 1
 */
// NOTE: This method could be made a method of the Node class. The best
//       architecture would be to make a subclass, HeadNode, of Node class and
//       including this method there. At the moment we incrementally create the
//       graph while identifying head nodes, we are using Node objects
//       everywhere. To follow the HeadNode subclass specialization route, we
//       either have to replace Node objects with HeadNode objects or do a first
//       pass through the input to identify head nodes and then create the graph.
void
AnalysisGraph::compute_fourier_coefficients_from_least_square_optimization(
                                           const Eigen::MatrixXd &sinusoidals,
                                           int n_components,
                                           vector<int> &head_node_ids) {
    int tot_sinusoidal_rows = 2 * n_components;

    for (auto hn_id: head_node_ids) {
        Node& hn = (*this)[hn_id];

        /* Setting up the linear system Ux = y to solve for the
         * Fourier coefficients using the least squares optimization */
        // Adding one additional column for cos(0) term
        Eigen::MatrixXd U = Eigen::MatrixXd::Zero(hn.tot_observations,
                                                  tot_sinusoidal_rows + 1);

        // Setting the coefficient for cos(0) term (α₀). In the traditional
        // Fourier decomposition this is 0.5 and when α₀ is used, we have to
        // divide it by 2.
        // To avoid this additional division, here we use 1 instead.
        U.col(0) = Eigen::VectorXd::Ones(hn.tot_observations); // ≡ cos(0)
        Eigen::VectorXd y = Eigen::VectorXd::Zero(hn.tot_observations);

        int row = 0;

        // Iterate through all the bins (partitions)
        for (auto [bin, data] : hn.partitioned_data) {

            // Iterate through all the observations in one bin
            for (double obs : data.second) {
                U.block(row, 1, 1, tot_sinusoidal_rows) =
                                      sinusoidals.block(bin, 0,
                                                        1, tot_sinusoidal_rows);
                y(row) = obs;
                row++;
            }
        }

        hn.fourier_coefficients =
            U.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
    }
}

/**
 * Assemble the LDS to generate all the head nodes with the same frequency. This
 * LDS is used to evolve the system to between bin midpoints to find the best
 * number of frequencies to be used to fit each head node.
 * @param A_sin_base: Base transition matrix to generate sinusoidals with all the
 *                   possible effective frequencies.
 * @param s0_sin: Initial state of the LDS that generates sinusoidal curves
 * @param period_freqs: All the effective frequencies that could be used to fit a
 *                     concept with the period common to concepts being modeled
 * @param fourier_coefficients: Should not be passed as a parameter. Should be
 *                              part of each Node. Fourier coefficients to fit
 *                              each concept.
 * @return pair (base transition matrix, initial state) for the LDS with the
 *         maximum effective size (with the maximum number of sinusoidals) to
 *         generate all the head nodes that share a common period
 *         0 radians is the initial angle.
 */
pair<Eigen::MatrixXd, Eigen::VectorXd>
AnalysisGraph::assemble_LDS_for_head_nodes_with_the_same_period(
                                const Eigen::MatrixXd &A_sin_base,
                                const Eigen::VectorXd &s0_sin,
                                const vector<double> &period_freqs,
                                int n_components,
                                unordered_map<int, int> &hn_to_mat_row,
                                const Eigen::MatrixXd &A_concept_base) {

    int num_verts = hn_to_mat_row.size();
    int tot_concept_rows = 2 * num_verts;
    int tot_sinusoidal_rows = 2 * n_components;
    int tot_rows = tot_concept_rows + tot_sinusoidal_rows;

    unordered_map<double, int> frequency_to_idx;
    for (int i = 0; i < n_components; i++) {
        frequency_to_idx[period_freqs[i]] = tot_concept_rows + 2 * i;
    }

    Eigen::MatrixXd A_concept_period_base = Eigen::MatrixXd::Zero(tot_rows,
                                                                  tot_rows);

    A_concept_period_base.topLeftCorner(tot_concept_rows,
                                        tot_concept_rows) = A_concept_base;

    A_concept_period_base.bottomRightCorner(tot_sinusoidal_rows,
                                            tot_sinusoidal_rows) =
             A_sin_base.topLeftCorner(tot_sinusoidal_rows, tot_sinusoidal_rows);

    Eigen::VectorXd s0_concept_period = Eigen::VectorXd::Zero(tot_rows);
    s0_concept_period.tail(tot_sinusoidal_rows) =
                                               s0_sin.head(tot_sinusoidal_rows);

    for (auto [hn_id, dot_row]: hn_to_mat_row) {
        Node& hn = (*this)[hn_id];

        int dot_dot_row = dot_row + 1;

        // Sinusoidal coefficient vector to calculate initial value for concept v
        Eigen::VectorXd v0 = Eigen::VectorXd::Zero(hn.fourier_coefficients.size());

        // Coefficient for cos(0) term (α₀). In the traditional Fourier
        // decomposition this to 0.5. When we compute α₀ we include this factor
        // into α₀ (Instead of computing α₀ as in the traditional Fourier series,
        // we compute α₀/2 straightaway).
        v0(0) = 1;

        for (int concept_freq_idx = 0; concept_freq_idx < hn.n_components;
             concept_freq_idx++) {

            double concept_freq = hn.fourier_freqs[concept_freq_idx]; // λω
            double concept_freq_squared = concept_freq * concept_freq;

            int concept_freq_idx_2 = concept_freq_idx * 2;
            int beta_omega_idx = concept_freq_idx_2 + 1;
            int alpha_omega_idx = concept_freq_idx_2 + 2;

            // Coefficient for sin(λω t) terms ≡ β_ω
            double beta_omega = hn.fourier_coefficients[beta_omega_idx];
            // Coefficient for λω cos(λω t) terms ≡ α_ω
            double alpha_omega = hn.fourier_coefficients[alpha_omega_idx];

            int sin_idx = frequency_to_idx[concept_freq];
            int cos_idx = sin_idx + 1;

            // Setting coefficients of the first derivative of the head node v
            // They are in row 2v in the transition matrix

            // Setting coefficients for sin terms: -freq^2 * cos_coefficient
            A_concept_period_base(dot_row, sin_idx) = -concept_freq_squared *
                                                 alpha_omega;

            // Setting coefficients for cos terms: sin_coefficient
            A_concept_period_base(dot_row, cos_idx) = beta_omega;

            // Setting coefficients of the second derivative of the head node v
            // They are in row 2 * v + 1 in the transition matrix

            // Setting coefficients for sin terms: -freq^2 * sin_coefficient
            A_concept_period_base(dot_dot_row, sin_idx) = -concept_freq_squared *
                                                     beta_omega;

            // Setting coefficients for cos terms: -freq^2 * cos_coefficient
            A_concept_period_base(dot_dot_row, cos_idx) = -concept_freq_squared *
                                                     alpha_omega;

            // Populating the sinusoidal coefficient vector to compute the
            // initial value for the concept v
            v0(beta_omega_idx) = s0_concept_period(sin_idx);
            v0(alpha_omega_idx) = s0_concept_period(cos_idx);
        }

        // Setting the initial value for concept v
        s0_concept_period(dot_row) = hn.fourier_coefficients.dot(v0);

        // Setting the initial derivative for concept v
        s0_concept_period(dot_dot_row) = A_concept_period_base.row(dot_row).dot(s0_concept_period);
    }

    return make_pair(A_concept_period_base, s0_concept_period);
}

bool AnalysisGraph::determine_the_best_number_of_components(
                                   const Eigen::MatrixXd &A_concept_period_base,
                                   const Eigen::VectorXd &s0_concept_period,
                                   int period,
                                   int n_components,
                                   unordered_map<int, int> &hn_to_mat_row) {

    int num_verts = hn_to_mat_row.size();

    Eigen::MatrixXd A_till_first_midpoint = (A_concept_period_base * 0.5).exp();
    Eigen::MatrixXd A_step = A_concept_period_base.exp();

    int lds_size = 2 * (num_verts + n_components);

    // Evolve the system for 1 period of time steps at between bin midpoints
    Eigen::MatrixXd preds = Eigen::MatrixXd::Zero(lds_size, period);
    preds.col(0) = A_till_first_midpoint * s0_concept_period;
//    preds.col(0) = s0_concept_period;

    for (int col = 1; col < period; col++) {
        preds.col(col) = A_step * preds.col(col - 1);
    }

    // Track whether the rmse for at least one concept got reduced for this
    // number of components. This means we have to check the rmse for
    // n_components. Monitoring this allows us to stop training when rmses are
    // not reducing for all the concepts with the specified period.
    bool rmses_are_reducing = false;

    for (auto [hn_id, v_preds_row]: hn_to_mat_row) {
        // NOTE: This for loop could be made a method of the Node class. The best
        //       architecture would be to make a subclass, HeadNode, of Node class and
        //       including this for loop as a method there. At the moment we incrementally create the
        //       graph while identifying head nodes, we are using Node objects
        //       everywhere. To follow the HeadNode subclass specialization route, we
        //       either have to replace Node objects with HeadNode objects or do a first
        //       pass through the input to identify head nodes and then create the graph.
        Node& hn = (*this)[hn_id];
        vector<double> errors;

        // TODO: Just for debugging delete
        //for (auto [midpoint, vals] : hn.partitioned_data) {
            //for (double val : vals.second) {

        for (auto [midpoint, vals] : hn.between_bin_midpoints) {
            for (double val : vals) {
                errors.push_back(val - preds(v_preds_row, midpoint));
            }
        }

        double rmse = sqrt(
            inner_product(errors.begin(), errors.end(), errors.begin(), 0.0)
                                                               / errors.size());

        cout << n_components << " : " << rmse << endl;
        if (hn.rmse_is_reducing) {
            hn.rmse_is_reducing = (rmse < hn.best_rmse);
            if (hn.rmse_is_reducing) {
                rmses_are_reducing = true;
                hn.best_rmse = rmse;
                hn.best_n_components = n_components;
                hn.best_fourier_coefficients = hn.fourier_coefficients;
            }
        }
    }
    return rmses_are_reducing;
}

void AnalysisGraph::check_sines(const Eigen::MatrixXd &A_sin_base,
                                const Eigen::VectorXd &s0_sin, int period) {

    CSVWriter writer("sines_" +
                     to_string(s0_sin.size() / 2) + ".csv");
    double step = 0.25;
    int tot_points = 2 * period * int(1/step) + 1;
    // Transition matrix to advance the sinusoidal generation LDS from one bin
    // to the next.
    //Eigen::MatrixXd A_sin_max_k_step = (A_sin_max_k_base * d_theta).exp();
    Eigen::MatrixXd A_sin_step = (A_sin_base * step).exp();

    // A matrix to accumulate sinusoidals of all the frequencies for all the bin
    // locations required for the Fourier curve fitting.
    Eigen::MatrixXd sinusoidals = Eigen::MatrixXd::Zero(s0_sin.size(), tot_points);
    sinusoidals.col(0) = s0_sin;

    // Evolve the sinusoidal generating LDS one step at a time for a whole
    // period to generate all the sinusoidal values required for the Fourier
    // reconstruction of the seasonal time series. Column j provides
    // sinusoidal values required for bin j.
    for (int col = 1; col < tot_points; col++) {
        sinusoidals.col(col) = A_sin_step * sinusoidals.col(col - 1);
    }

    for (int row = 0; row < s0_sin.size(); row++) {
        vector<double> row_vals(tot_points);
        Eigen::VectorXd::Map(&row_vals[0], tot_points) = sinusoidals.row(row);
        writer.write_row(row_vals.begin(), row_vals.end());
    }
}

// 1090
void
AnalysisGraph::fit_seasonal_head_node_model_via_fourier_decomposition() {

    // TODO: Just for debugging delete
    this->set_indicator_means_and_standard_deviations();

    // Group seasonal head nodes according to their seasonality.
    unordered_map<int, vector<int>> period_to_head_nodes;

    for (int head_node_id: this->head_nodes) {
        Node& hn = (*this)[head_node_id];

        // TODO: Just for debugging delete
        /*
        hn.period = 8;
        hn.tot_observations = 24; // Total observations for a concept
        hn.partitioned_data =
            {{0, {{},{-0.2, 0, 0.2}}},
             {1, {{},{1.8, 2, 2.2}}},
             {2, {{},{5.8, 6, 6.2}}},
             {3, {{},{3.8, 4, 4.2}}},
             {4, {{},{2.8, 3, 3.2}}},
             {5, {{},{7.8, 8, 8.2}}},
             {6, {{},{4.8, 5, 5.2}}},
             {7, {{},{3.8, 4, 4.2}}}
            };
        hn.between_bin_midpoints =
            {{0, {1, 1.2, 0.8}},
             {1, {4, 4.2, 3.8}},
             {2, {5, 5.2, 4.8}},
             {3, {3.5, 3.7, 3.3}},
             {4, {5.5, 5.7, 5.3}},
             {5, {6.5, 6.7, 6.3}},
             {6, {4.5, 4.7, 4.3}},
             {7, {2.1, 2}}
            };
        */
//        {{0, {0, 0, 0}},
//         {1, {2, 2, 2}},
//         {2, {6, 6, 6}},
//         {3, {4, 4, 4}},
//         {4, {3, 3, 3}},
//         {5, {8, 8, 8}},
//         {6, {5, 5, 5}},
//         {7, {4, 4, 4}}
//        };
        /*
        unordered_map<int, vector<double>> between_bin_midpoints2 =
            {{0, {-0.2, 0, 0.2}},
             {1, {1.8, 2, 2.2}},
             {2, {5.8, 6, 6.2}},
             {3, {3.8, 4, 4.2}},
             {4, {2.8, 3, 3.2}},
             {5, {7.8, 8, 8.2}},
             {6, {4.8, 5, 5.2}},
             {7, {3.8, 4, 4.2}}
            };
        unordered_map<int, vector<double>> between_bin_midpoints1 =
            {{0, {0, 0, 0}},
             {1, {2, 2, 2}},
             {2, {6, 6, 6}},
             {3, {4, 4, 4}},
             {4, {3, 3, 3}},
             {5, {8, 8, 8}},
             {6, {5, 5, 5}},
             {7, {4, 4, 4}}
            };
        unordered_map<int, vector<double>> between_bin_midpoints =
            {{0, {1, 1.2, 0.8}},
             {1, {4, 4.2, 3.8}},
             {2, {5, 5.2, 4.8}},
             {3, {3.5, 3.7, 3.3}},
             {4, {5.5, 5.7, 5.3}},
             {5, {6.5, 6.7, 6.3}},
             {6, {4.5, 4.7, 4.3}},
             {7, {2.1, 2}}
            };
        // Bin i refer to the midpoint between bin i and (i+1) % period
        unordered_map<double, vector<double>> between_bin_mid_point_linear_interpolated_values =
            {{0.5, {1, 1.2, 0.8}},
             {1.5, {4, 4.2, 3.8}},
             {2.5, {5, 5.2, 4.8}},
             {3.5, {3.5, 3.7, 3.3}},
             {4.5, {5.5, 5.7, 5.3}},
             {5.5, {6.5, 6.7, 6.3}},
             {6.5, {4.5, 4.7, 4.3}},
             {7.5, {2.1, 2}}
            };
        */

        period_to_head_nodes[hn.period].push_back(head_node_id);
    }

    // TODO: Just for debugging delete
    for (auto [p, hn_ids]: period_to_head_nodes) {
        for (auto hn_id: hn_ids) {
            Node& hn = (*this)[hn_id];
            cout << p << " - " << hn.name << " - " << hn.tot_observations << "\n";
        }
    }

    std::unordered_set<double> fourier_frequency_set;

    for (auto [period, hn_ids]: period_to_head_nodes) {
        //Node& hn = (*this)[hn_ids[0]];  // TODO: Just for debugging delete

        // The maximum number of components we are going to evaluate in search
        // of the best number of components to be used.
        // max_k < period / 2 (Nyquist theorem)
        // TODO: Just for debugging. Change to period / 2 // Integer division
        int max_k = period / 2;

        // Generate the maximum number of sinusoidal frequencies needed for the
        // period. By the Nyquist theorem this number is floor(period / 2)
        // The actual number of frequencies used to model a concept could be
        // less than this, which is decided by computing the root mean squared
        // error of the predictions for each number of components
        vector<double> period_freqs =
                           this->generate_frequencies_for_period(max_k, period);

        // Assemble the sinusoidal generating LDS with the maximum number of
        // components. This includes all the sinusoidals needed for every lesser
        // number of components we are going to try out.
        auto [A_sin_max_k_base, s0_sin_max_k] =
            this->assemble_sinusoidal_generating_LDS(period_freqs);

        Eigen::MatrixXd sinusoidals = this->generate_sinusoidal_values_for_bins(
            A_sin_max_k_base, s0_sin_max_k, period);

        // Assign transition matrix rows to head nodes with this period.
        // NOTE: At this moment we do not need to reformat the head node vector
        //       this way. We could just use the head node vector as it is and
        //       compute the transition matrix row based on the index at which
        //       each head node is at.
        //       I am doing this thinking to reuse the
        //       assemble_LDS_for_head_nodes_with_the_same_period() method to
        //       assemble the final LDS with all the nodes with body nodes
        //       included. At the moment, in that system, head nodes does not
        //       occupy a contiguous range of rows in the transition matrix.
        unordered_map<int, int> hn_to_mat_row;
        for (int v = 0; v < hn_ids.size(); v++) {
            hn_to_mat_row[hn_ids[v]] = 2 * v;

            // TODO: Just for debugging.
            cout << "node id: " << hn_ids[v] << " - mat row: " << 2 * v << "\n";

            // Again in the light of reusing the LDS assembly code to assemble
            // the full matrix
            Node& hn = (*this)[hn_ids[v]];
            hn.fourier_freqs = period_freqs;
        }

        // Again creating a dummy zero matrix to make
        // assemble_LDS_for_head_nodes_with_the_same_period() method more general
        int n_concept_rows = 2 * hn_ids.size();
        Eigen::MatrixXd A_concept_base_dummy = Eigen::MatrixXd::Zero(
                                                                n_concept_rows,
                                                                n_concept_rows);

        for (int components = 0; components <= max_k; components++) {
            for (auto hn_id: hn_ids) {
                // Again in the light of reusing the LDS assembly code to assemble
                // the full matrix
                Node& hn = (*this)[hn_id];
                hn.n_components = components;
            }

            this->compute_fourier_coefficients_from_least_square_optimization(
                                            sinusoidals, components, hn_ids);

            // Assemble the LDS that generates the head nodes with this
            // period using this number of components.
            auto [A_concept_period_base, s0_concept_period] =
                this->assemble_LDS_for_head_nodes_with_the_same_period(
                                                          A_sin_max_k_base,
                                                          s0_sin_max_k,
                                                          period_freqs,
                                                          components,
                                                          hn_to_mat_row,
                                                          A_concept_base_dummy);

            bool rmses_are_reducing = determine_the_best_number_of_components(
                                                          A_concept_period_base,
                                                          s0_concept_period,
                                                          period,
                                                          components,
                                                       hn_to_mat_row);

            if (!rmses_are_reducing) {
                break;
            }
        }

        Node& hn_dbg = (*this)[hn_ids[0]];  // TODO: Just for debugging delete
        cout << "Best: " << hn_dbg.best_n_components << " : " << hn_dbg.best_rmse << endl;

        // Accumulate all the fourier frequencies needed to model all the head
        // nodes with this period
        int max_n_components_for_period = 0;
        for (auto hn_id: hn_ids) {
            Node& hn = (*this)[hn_id];

            // In the assemble_LDS_for_head_nodes_with_the_same_period() method
            // Node object members: n_components and fourier_coefficients are
            // accessed.
            // When we assemble the final transition matrix with all the fitted
            // seasonal head nodes and all the body nodes, we have to use Node
            // object members: best_n_components and best_fourier_coefficients,
            // for the final system to utilize the fitted seasonal models.
            // Therefore, we reassign best_n_components and
            // best_fourier_coefficient to n_components and
            // fourier_coefficients members so that we could reuse the
            // assemble_LDS_for_head_nodes_with_the_same_period() method
            // seamlessly without any change or state checking at the time of
            // assembling the transition matrix for the final complete system.
            hn.n_components = hn.best_n_components;
            hn.fourier_coefficients = hn.best_fourier_coefficients;
            hn.best_fourier_coefficients.resize(0);  // To save some memory

            if (hn.best_n_components > max_n_components_for_period) {
                max_n_components_for_period = hn.best_n_components;
            }
        }
        fourier_frequency_set.insert(period_freqs.begin(), period_freqs.begin() + max_n_components_for_period);
    }

    // TODO: Just for debugging. Move to transition matrix assembly pari in
    // sampling.cpp to include all the body nodes as well. For the moment testing
    // Assemble the final LDS with all the variables
    vector<double> all_freqs(fourier_frequency_set.begin(), fourier_frequency_set.end());
    sort(all_freqs.begin(), all_freqs.end());

    auto [A_sin_all_base, s0_sin_all] =
        this->assemble_sinusoidal_generating_LDS(all_freqs);


    // TODO: This should change to actual head node ids when assembling with all concepts
    unordered_map<int, int> hn_to_mat_row;
    int row = 0;
    for (int hn_id: this->head_nodes) {
        hn_to_mat_row[hn_id] = row;
        row += 2;
    }

    // TODO: Should be the actual transition matrix without sinusoidal part
    int n_concept_rows = 2 * this->head_nodes.size();
    Eigen::MatrixXd A_concept_base_dummy = Eigen::MatrixXd::Zero(
        n_concept_rows,
        n_concept_rows);

    auto [A_concept_full_base, s0_concept_full] =
        this->assemble_LDS_for_head_nodes_with_the_same_period(
            A_sin_all_base,
            s0_sin_all,
            all_freqs,
            all_freqs.size(),
            hn_to_mat_row,
            A_concept_base_dummy);

    cout << A_concept_full_base << "\n";

    this->check_sines(A_concept_full_base, s0_concept_full, 12);
}
