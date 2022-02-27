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
    // one cycle = period radians
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
   *         0 radians in the initial angle.
 */
pair<Eigen::MatrixXd, Eigen::VectorXd>
AnalysisGraph::assemble_sinusoidal_generating_LDS(const vector<double> &freqs) {
    unsigned short comps_2 = freqs.size() * 2;
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
   * @param s0_sin: Initial state (0 radians0) for sinusoidal generating LDS
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
    // Transition matrix to advance the sinusoidal generation LDS from one bin
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
Eigen::VectorXd
AnalysisGraph::compute_fourier_coefficients_from_least_square_optimization(
                                           const Eigen::MatrixXd &sinusoidals) {
    int tot_observations = 24; // Total observations for a concept
    unordered_map<int, pair<vector<int>, vector<double>>> partitioned_data =
        {{0, {{},{-0.2, 0, 0.2}}},
         {1, {{},{1.8, 2, 2.2}}},
         {2, {{},{5.8, 6, 6.2}}},
         {3, {{},{3.8, 4, 4.2}}},
         {4, {{},{2.8, 3, 3.2}}},
         {5, {{},{7.8, 8, 8.2}}},
         {6, {{},{4.8, 5, 5.2}}},
         {7, {{},{3.8, 4, 4.2}}}
        };

    /* Setting up the linear system Ux = y to solve for the
     * Fourier coefficients using the least squares optimization */
    // Adding one additional column for cos(0) term
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(tot_observations, sinusoidals.cols() + 1);
    // Setting the coefficient for cos(0) term (α₀). In the traditional Fourier
    // decomposition this is 0.5 and when α₀ is used, we have to divide it by 2.
    // To avoid this additional division, here we use 1 instead.
    U.col(0) = Eigen::VectorXd::Ones(tot_observations); // ≡ cos(0)
    Eigen::VectorXd y = Eigen::VectorXd::Zero(tot_observations);

    unsigned int row = 0;

    // Iterate through all the bins (partitions)
    for (auto [bin, data]: partitioned_data) {

        // Iterate through all the observations in one bin
        for (double obs: data.second) {
            U.block(row, 1, 1, sinusoidals.cols()) =
                                                           sinusoidals.row(bin);
            y(row) = obs;
            row++;
        }
    }

    Eigen::VectorXd fourier_coefficients =
        U.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

    return fourier_coefficients;
}

void AnalysisGraph::check_sines(const Eigen::MatrixXd &A_sin_base,
                                const Eigen::VectorXd &s0_sin, int period) {

    CSVWriter writer("sines.csv");
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
    std::unordered_set<double> frequency_set;

    int period = 8;
    // The maximum number of components we are going to evaluate in search of
    // the best number of components to be used.
    // max_k < period / 2 (Nyquist theorem)
    int max_k = 4;
    int k = 2;

    // Generate the maximum number of sinusoidal frequencies needed for the
    // period. By the Nyquist theorem this number is floor(period / 2)
    // The actual number of frequencies used to model a concept could be less
    // than this, which is decided by computing the root mean
    // squared error of the predictions for each number of components
    vector<double> period_freqs = this->generate_frequencies_for_period(max_k,
                                                                        period);

    // Assemble the sinusoidal generating LDS with the maximum number of
    // components. This includes all the sinusoidals needed for every lesser
    // number of components we are going to try out.
    auto [A_sin_max_k_base, s0_sin_max_k] =
                this->assemble_sinusoidal_generating_LDS(period_freqs);

    Eigen::MatrixXd sinusoidals = this->generate_sinusoidal_values_for_bins(
                                                               A_sin_max_k_base,
                                                               s0_sin_max_k,
                                                               period);

    Eigen::VectorXd fourier_coefficients =
        this->compute_fourier_coefficients_from_least_square_optimization(
                                                                   sinusoidals);

    // By evaluating the root mean squared error on the validation set, decide
    // the frequencies to be used to model this concept
    vector<double> concept_freqs = period_freqs;

    // After deciding the best number of components for each concept, we add
    // them to a global set of frequencies needed
    frequency_set.insert(concept_freqs.begin(), concept_freqs.end());

    int num_verts = 1; // The number of vertices in the CAG

    // Once we are done with deciding all the frequencies required for the
    // system, we create the frequency to transition matrix index map
    vector<double> frequency_vec(frequency_set.begin(), frequency_set.end());
    sort(frequency_vec.begin(), frequency_vec.end());
    unordered_map<double, int> frequency_to_idx;
    for (int i = 0; i < frequency_vec.size(); i++) {
        frequency_to_idx[frequency_vec[i]] = 2 * (num_verts + i);
    }

    int tot_components = max_k;  // This should change to all the components for all the head nodes
    int tot_rows = 2 * (num_verts + tot_components);

    this->A_original = Eigen::MatrixXd::Zero(tot_rows, tot_rows);
    this->A_original.bottomRightCorner(2 * tot_components,
                                       2 * tot_components) = A_sin_max_k_base;

    this->s0 = Eigen::VectorXd::Zero(tot_rows);
    this->s0.tail(2 * tot_components) = s0_sin_max_k;


    for (int v = 0; v < num_verts; v++) {

        int dot_row = 2 * v;
        int dot_dot_row = dot_row + 1;

        // Sinusoidal coefficient vector to calculate initial value for concept v
        Eigen::VectorXd v0 = Eigen::VectorXd::Zero(fourier_coefficients.size());
        // Coefficient for cos(0) term (α₀). In the traditional Fourier
        // decomposition this to 0.5. When we compute α₀ we include this factor
        // into α₀ (Instead of computing α₀ as in the traditional Fourier series,
        // we compute α₀/2 straightaway).
        v0(0) = 1;

        for (int concept_freq_idx = 0; concept_freq_idx < concept_freqs.size();
             concept_freq_idx++) {

            double concept_freq = concept_freqs[concept_freq_idx]; // λω
            double concept_freq_squared = concept_freq * concept_freq;

            int concept_freq_idx_2 = concept_freq_idx * 2;
            int beta_omega_idx = concept_freq_idx_2 + 1;
            int alpha_omega_idx = concept_freq_idx_2 + 2;

            // Coefficient for sin(λω t) terms ≡ β_ω
            double beta_omega = fourier_coefficients[beta_omega_idx];
            // Coefficient for λω cos(λω t) terms ≡ α_ω
            double alpha_omega = fourier_coefficients[alpha_omega_idx];

            int sin_idx = frequency_to_idx[concept_freq];
            int cos_idx = sin_idx + 1;

            // Setting coefficients of the first derivative of the head node v
            // They are in row 2v in the transition matrix

            // Setting coefficients for sin terms: -freq^2 * cos_coefficient
            this->A_original(dot_row, sin_idx) = -concept_freq_squared *
                                                 alpha_omega;

            // Setting coefficients for cos terms: sin_coefficient
            this->A_original(dot_row, cos_idx) = beta_omega;

            // Setting coefficients of the second derivative of the head node v
            // They are in row 2 * v + 1 in the transition matrix

            // Setting coefficients for sin terms: -freq^2 * sin_coefficient
            this->A_original(dot_dot_row, sin_idx) = -concept_freq_squared *
                                                     beta_omega;

            // Setting coefficients for cos terms: -freq^2 * cos_coefficient
            this->A_original(dot_dot_row, cos_idx) = -concept_freq_squared *
                                                     alpha_omega;

            // Populating the sinusoidal coefficient vector to compute the
            // initial value for the concept v
            v0(beta_omega_idx) = this->s0(sin_idx);
            v0(alpha_omega_idx) = this->s0(cos_idx);
        }

        // Setting the initial value for concept v
        this->s0(dot_row) = fourier_coefficients.dot(v0);

        // Setting the initial derivative for concept v
        this->s0(dot_dot_row) = this->A_original.row(dot_row).dot(this->s0);
    }

    this->check_sines(this->A_original, this->s0, period);
}
