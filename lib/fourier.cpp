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
   * @param n_components: The number of pure sinusoidal frequencies to generate
   * @param period: The period of the variable(s) being modeled by Fourier
   *                decomposition. All variable(s) share the same period.
   * @return A vector of all the effective frequencies
 */
vector<double> AnalysisGraph::generate_frequencies_for_period(int period,
                                                              int n_components) {
    vector<double> freqs(n_components);

    // λ is the amount we have to stretch/shrink pure sinusoidal curves to make
    // one sine cycle = period radians
    double lambda = 2.0 * M_PI / period;

    // ω is the frequency of each stretched/shrunk sinusoidal curve.
    // With respect to trigonometric function differentiation, the effective
    // frequency of pure sinusoidal curves is λω
    for (int omega = 1; omega <= n_components; omega++) {
        freqs[omega - 1] = lambda * omega;
    }

    return freqs;
}

/**
   * Assemble the LDS to generate sinusoidals of desired effective frequencies
   * @param freqs: A vector of effective frequencies
   *               (2πω / period; ω = 1, 2, ...)
   * @return pair (base transition matrix, initial state)
   *         0 radians is the initial angle.
 */
pair<Eigen::MatrixXd, Eigen::VectorXd>
AnalysisGraph::assemble_sinusoidal_generating_LDS(const vector<double> &freqs) {
    int lds_size = freqs.size() * 2;

    // For each effective frequency, there are two rows in the transition matrix:
    //      1) for first derivative and
    //      2) for second derivative
    // In the state vector these rows are:
    //      1) the value
    //      2) the first derivative---
    Eigen::MatrixXd A_sin = Eigen::MatrixXd::Zero(lds_size, lds_size);
    Eigen::VectorXd s0_sin = Eigen::VectorXd::Zero(lds_size);

    for (int i = 0; i < freqs.size(); i++) {
        int i2 = i * 2;     // first derivative matrix row
        int i2p1 = i2 + 1;  // second derivative matrix row

        // Assembling a block diagonal matrix
        // with each 2 x 2 block having the format
        // columns ->   2i        2i+1
        //            _________________
        // row 2i    |   0          1 |
        // row 2i+1  | -(λω)^2      0 |
        // ω = i+1 & λ = 2π/period
        A_sin(i2, i2p1) = 1;
        A_sin(i2p1, i2) = -freqs[i] * freqs[i];

        // Considering t₀ = 0 radians, the initial state of the sinusoidal
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
 * Evolves the provided LDS (A_base and _s0) n_modeling_time_steps (e.g. months)
 * taking step_size steps each time. For example, if the step_size is 0.25,
 * there will be 4 prediction points per one modeling time step.
 * @param A_base: Base transition matrix that define the LDS.
 * @param _s0: Initial state of the system. Used _s0 instead of s0 because s0 is
 *             the member variable that represent the initial state of the final
 *             system that is used by the AnalysisGraph object.
 * @param n_modeling_time_steps: The number of modeling time steps (full time
 *                               steps, e.g. months) to evolve the system.
 * @param step_size: Amount to advance the system at each step.
 * @return A matrix of evolved values. Each column has values for one step.
 *              row 2i   - Values for variable i in the system
 *              row 2i+1 - Derivatives for variable i in the system
 */
Eigen::MatrixXd AnalysisGraph::evolve_LDS(const Eigen::MatrixXd &A_base,
                                          const Eigen::VectorXd &_s0,
                                          int n_modeling_time_steps,
                                          double step_size) {
    int tot_steps = int(n_modeling_time_steps / step_size);
    int lds_size = _s0.size();

    // Transition matrix to advance the system one step_size forward
    Eigen::MatrixXd A_step = (A_base * step_size).exp();

    // A matrix to accumulate predictions of the system
    Eigen::MatrixXd preds = Eigen::MatrixXd::Zero(lds_size, tot_steps);
    preds.col(0) = _s0;

    // Evolve the LDS one step_size at a time for desired number of steps
    for (int col = 1; col < tot_steps; col++) {
        preds.col(col) = A_step * preds.col(col - 1);
    }

    return preds;
}

/**
   * Generate a matrix of sinusoidal values of all the desired effective
   * frequencies for all the bin locations.
   * @param A_sin_base: Base transition matrix for sinusoidal generating LDS
   * @param s0_sin: Initial state (0 radians) for sinusoidal generating LDS
   * @param period: Period shared by the time series that will be fitted using
   *                the generated sinusoidal values. This period must be the
   *                same period used to generate the vector of effective
   *                frequencies used to assemble the transition matrix
   *                A_sin_base and the initial state s0_sin.
   * @return A matrix of required sinusoidal values.
   *         row t contains sinusoidals for bin t (radians)
   *         col 2ω contains sin(λω t)
   *         col 2ω+1 contains λω cos(λω t)
   *
   *         with ω = 1, 2, ... & λ = 2π/period
 */
Eigen::MatrixXd
AnalysisGraph::generate_sinusoidal_values_for_bins(
                                              const Eigen::MatrixXd &A_sin_base,
                                              const Eigen::VectorXd &s0_sin,
                                              int period) {
    // Evolve the sinusoidal generating LDS one step at a time for a whole
    // period to generate all the sinusoidal values required for the Fourier
    // reconstruction of the seasonal time series. Column t provides
    // sinusoidal values required for bin t.
    Eigen::MatrixXd sinusoidals = this->evolve_LDS(A_sin_base, s0_sin, period,
                                                                             1);

    // Transpose the sinusoidal matrix so that row t contains the sinusoidal
    // values for bin t.
    sinusoidals.transposeInPlace();

    return sinusoidals;
}

// NOTE: This method could be made a method of the Node class. The best
//       architecture would be to make a subclass, HeadNode, of Node class and
//       include this method there. At the moment we incrementally create the
//       graph while identifying head nodes, we are using Node objects
//       everywhere. To follow the HeadNode subclass specialization route, we
//       either have to replace Node objects with HeadNode objects or do a first
//       pass through the input to identify head nodes and then create the graph.
/**
 * For each head node, computes the Fourier coefficients to fit a seasonal
 * curve to partitioned observations (bins) using the least square
 * optimization.
 * @param sinusoidals: Sinusoidal values of required effective frequencies at
 *                     each bin position. Row b contains all the sinusoidal
 *                     values for bin b.
 *                        sinusoidals(b, 2(ω-1))     =    sin(λω b)
 *                        sinusoidals(b, 2(ω-1) + 1) = λω cos(λω b)
 *                    with ω = 1, 2, ... & λ = 2π/period
 * @param n_components: The number of different sinusoidal frequencies used to
 *                      fit the seasonal head node model. The supplied
 *                      sinusoidals matrix could have sinusoidal values for
 *                      higher frequencies than needed
 *                      (number of columns > 2 * n_components). This method
 *                      utilizes only the first 2 * n_components columns.
 * @param head_node_ids: A list of head nodes with the period matching the
 *                       period represented in the provided sinusoidals. The
 *                       period of all the head nodes in this list must be the
 *                       same as the period parameter used when generating the
 *                       sinusoidals.
 * @return The Fourier coefficients in the order: α₀, β₁, α₁, β₂, α₂, ...
 *         α₀ is the coefficient for    cos(0)/2  term
 *         αᵢ is the coefficient for λi cos(λi b) term
 *         βᵢ is the coefficient for    sin(λi b) term
 *
 *         with i = 1, 2, ... & λ = 2π/period & b = 0, 1, ..., period - 1
 */
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
                // Only the first tot_sinusoidal_rows columns of the
                // sinusoidals are used.
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
  * Assembles the LDS to generate the head nodes specified in the hn_to_mat_row
  * map by Fourier reconstruction using the sinusoidal frequencies generated by
  * the provided sinusoidal generating LDS: A_sin_base and s0_sin.
  * @param A_sin_base: Base transition matrix to generate sinusoidals with all
  *                    the possible effective frequencies.
  * @param s0_sin: Initial state of the LDS that generates sinusoidal curves.
  * @param n_components: The number of different sinusoidal frequencies used to
  *                      fit the seasonal head nodes. The sinusoidal generating
  *                      LDS provided (A_sin_base, s0_sin) could generate more
  *                      sinusoidals of higher frequencies. This method uses
  *                      only the lowest n_components frequencies to assemble
  *                      the complete system.
  * @param hn_to_mat_row: A map that maps each head node being modeled to the
  *                       transition matrix rows and state vector rows.
  *                       Each concept is allocated to two consecutive rows.
  *                       In the transition matrix:
  *                             even row) for first derivative
  *                             odd  row) for second derivative
  *                       In the state vector:
  *                             even row) the value
  *                             odd  row) for first derivative and
  *                       This map indicates the even row numbers each concept
  *                       is assigned to.
  * @param A_concept_base: Base transition matrix that models the relationships
  *                        between concepts that are being modeled by the
  *                        assembled LDS.
  * @return pair (base transition matrix, initial state) for the complete LDS
  *         with the specified number of sinusoidal frequencies (n_components)
  *         0 radians is the initial angle.
  */
pair<Eigen::MatrixXd, Eigen::VectorXd>
AnalysisGraph::assemble_head_node_modeling_LDS(
                                       const Eigen::MatrixXd &A_sin_base,
                                       const Eigen::VectorXd &s0_sin,
                                       int n_components,
                                       unordered_map<int, int> &hn_to_mat_row) {

    int n_concepts = hn_to_mat_row.size(); // TODO: This will not work when we are assembling the LDS with body nodes as well. Have to pass n_concepts separately.
    int tot_concept_rows = 2 * n_concepts;
    int tot_sinusoidal_rows = 2 * n_components;
    int lds_size = tot_concept_rows + tot_sinusoidal_rows;

    unordered_map<double, int> frequency_to_idx;
    for (int i = 0; i < tot_sinusoidal_rows; i += 2) {
        // Since we are using t₀ = 0 radians as the initial time point, the
        // odd rows of the initial state of the sinusoidal generating LDS,
        // s0_sin, contains all the effective frequencies (λω) used to generate
        // the sinusoidal curves of different effective frequencies. Here we
        // are extracting those and assign them to the rows of the complete LDS.
        // The first tot_concept_rows of the system are used to model the actual
        // concepts. The latter tot_sinusoidal_rows are used to generate the
        // sinusoidal curves of all the effective frequencies used to model the
        // seasonal head nodes.
        frequency_to_idx[s0_sin(i + 1)] = tot_concept_rows + i;
    }

    Eigen::MatrixXd A_complete_base = Eigen::MatrixXd::Zero(lds_size, lds_size);

    // TODO: We do not have to do this within this method, when assembling the
    // final LDS, we cold do this to the matrix and initial state returned by
    // this method.
    //A_complete_base.topLeftCorner(tot_concept_rows, tot_concept_rows) =
    //                                                             A_concept_base;

    A_complete_base.bottomRightCorner(tot_sinusoidal_rows, tot_sinusoidal_rows)
           = A_sin_base.topLeftCorner(tot_sinusoidal_rows, tot_sinusoidal_rows);

    Eigen::VectorXd s0_complete = Eigen::VectorXd::Zero(lds_size);
    s0_complete.tail(tot_sinusoidal_rows) = s0_sin.head(tot_sinusoidal_rows);

    for (auto [hn_id, dot_row]: hn_to_mat_row) {
        Node& hn = (*this)[hn_id];

        int dot_dot_row = dot_row + 1;

        // Sinusoidal coefficient vector to calculate initial value for concept v
        Eigen::VectorXd v0 = Eigen::VectorXd::Zero(hn.fourier_coefficients.size());

        // Coefficient for cos(0) term (α₀). In the traditional Fourier
        // decomposition this is 0.5. When we compute α₀ we include this factor
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

            // Setting coefficients of the first derivative of the head node
            // hn_id. They are in row 2 * hn_id in the transition matrix.

            // Setting coefficients for sin terms: -freq^2 * cos_coefficient
            A_complete_base(dot_row, sin_idx) = -concept_freq_squared *
                                                 alpha_omega;

            // Setting coefficients for cos terms: sin_coefficient
            A_complete_base(dot_row, cos_idx) = beta_omega;

            // Setting coefficients of the second derivative of the head node
            // hn_id. They are in row 2 * hn_id + 1 in the transition matrix.

            // Setting coefficients for sin terms: -freq^2 * sin_coefficient
            A_complete_base(dot_dot_row, sin_idx) = -concept_freq_squared *
                                                     beta_omega;

            // Setting coefficients for cos terms: -freq^2 * cos_coefficient
            A_complete_base(dot_dot_row, cos_idx) = -concept_freq_squared *
                                                     alpha_omega;

            // Populating the sinusoidal coefficient vector to compute the
            // initial value for the head node hn_id.
            v0(beta_omega_idx) = s0_complete(sin_idx);
            v0(alpha_omega_idx) = s0_complete(cos_idx);
        }

        // Setting the initial value for head node hn_id.
        s0_complete(dot_row) = hn.fourier_coefficients.dot(v0);

        // Setting the initial derivative for head node hn_id.
        s0_complete(dot_dot_row) = A_complete_base.row(dot_row).dot(s0_complete);
    }

    return make_pair(A_complete_base, s0_complete);
}

/**
 *
 * @param A_hn_period_base: Transition matrix for the LDS that models seasonal
 *                          head nodes with the same period.
 * @param s0_hn_period: Initial state for the LDS modeling seasonal head nodes
 *                      with the same period. t₀ = 0 radians.
 * @param period: Period of the seasonal head nodes modeled by the LDS defined
 *                by A_hn_period_base and s0_hn_period.
 * @param n_components: Total number of sinusoidal frequencies used to model
 *                      all the seasonal head nodes in this LDS.
 * @param hn_to_mat_row: A map that maps each head node being modeled to the
 *                       transition matrix rows and state vector rows.
 *                       Each concept is allocated to two consecutive rows.
 *                       In the transition matrix:
 *                             even row) for first derivative
 *                             odd  row) for second derivative
 *                       In the state vector:
 *                             even row) the value
 *                             odd  row) for first derivative
 *                       This map indicates the even row numbers each concept
 *                       is assigned to.
 * @return A boolean value indicating whether the Fourier decomposition based
 *         seasonal head node model got improved for any of the head nodes
 *         specified in hn_to_mat_row.
 *              true  ⇒ Got improved. Should check for n_components + 1
 *              false ⇒ Did not improve. No point in checking for
 *                      n_components + 1. Stop early.
 */
bool AnalysisGraph::determine_the_best_number_of_components(
                                   const Eigen::MatrixXd & A_hn_period_base,
                                   const Eigen::VectorXd & s0_hn_period,
                                   int period,
                                   int n_components,
                                   unordered_map<int, int> &hn_to_mat_row) {

    Eigen::MatrixXd A_till_first_midpoint = (A_hn_period_base * 0.5).exp();

    // Evolve the system for 1 period of time steps at between bin midpoints.
    // Column b of preds contain the predictions for the midpoint between bin
    // b and bin (b + 1) % period.
    Eigen::MatrixXd preds = this->evolve_LDS(A_hn_period_base,
                                             A_till_first_midpoint *
                                                                   s0_hn_period,
                                             period, 1);

    // Track whether the rmse for at least one head node got reduced for this
    // number of components. This means we have to check the rmse for
    // n_components + 1. Monitoring this allows us to stop training when rmses
    // are not reducing for all the head nodes with the specified period.
    bool rmses_are_reducing = false;

    for (auto [hn_id, hn_preds_row]: hn_to_mat_row) {
        // NOTE: This for loop could be made a method of the Node class. The
        //       best architecture would be to make a subclass, HeadNode, of
        //       Node class and include this for loop as a method there. At the
        //       moment we incrementally create the graph while identifying head
        //       nodes, we are using Node objects everywhere. To follow the
        //       HeadNode subclass specialization route, we either have to
        //       replace Node objects with HeadNode objects or do a first pass
        //       through the input to identify head nodes and then create the
        //       graph.
        Node& hn = (*this)[hn_id];
        vector<double> errors;

        // TODO: Just for debugging delete
        //for (auto [midpoint, vals] : hn.partitioned_data) {
            //for (double val : vals.second) {

        // Iterate through all the midpoint bins
        for (auto [midpoint, vals] : hn.between_bin_midpoints) {
            // Iterate through all the linear interpolated midpoints in the
            // current midpoint bin.
            for (double val : vals) {
                errors.push_back(val - preds(hn_preds_row, midpoint));
            }
        }

        double rmse = sqrt(
            inner_product(errors.begin(), errors.end(), errors.begin(), 0.0)
                                                               / errors.size());

        cout << n_components << " : " << rmse << endl;
        if (hn.rmse_is_reducing) {
            // RMSE for this head node got reduced for this head node for
            // n_components - 1 components.
            hn.rmse_is_reducing = (rmse < hn.best_rmse);
            if (hn.rmse_is_reducing) {
                // RMSE for this head node got reduced this time as well for
                // n_components components.

                // Remember that rmse for at least one head node got reduced
                // during this run. So, we need to check again for
                // n_components + 1 components.
                rmses_are_reducing = true;

                // Remember the best model we have so far.
                hn.best_rmse = rmse;
                hn.best_n_components = n_components;
                hn.best_fourier_coefficients = hn.fourier_coefficients;
            }
        }
    }
    return rmses_are_reducing;
}

/**
 *
 * @param A_base: Base transition matrix that define the LDS.
 * @param _s0: Initial state of the system (t₀ = 0 radians). Used _s0 instead of
 *             s0 because s0 is the member variable that represent the initial
 *             state of the final system that is used by the AnalysisGraph
 *             object.
 * @param n_time_steps: The number of modeling time steps (full time steps,
 *                      e.g. months) to evolve the system.
 */
void AnalysisGraph::predictions_to_csv(const Eigen::MatrixXd &A_base,
                                       const Eigen::VectorXd &_s0,
                                       int n_time_steps) {

    CSVWriter writer("head_node_predictions_" +
                     to_string(_s0.size() / 2) + ".csv");
    int lds_size = _s0.size();

    // Evolve the LDS one step_size at a time for desired number of steps
    Eigen::MatrixXd preds = this->evolve_LDS(A_base, _s0, n_time_steps, 0.25);

    int tot_steps = preds.cols();

    // Transpose the prediction matrix so that
    //      col 2i   - Predictions for variable i in the system
    //      col 2i+1 - Derivatives for variable i in the system
    // Each row is a time step
    // This has tot_step rows and lds_size columns.
    preds.transposeInPlace();

    // Output the whole prediction matrix, including derivatives, into a csv
    // file.
    for (int step = 0; step < tot_steps; step++) {
        vector<double> preds_at_step(lds_size);
        Eigen::VectorXd::Map(&preds_at_step[0], lds_size) = preds.row(step);
        writer.write_row(preds_at_step.begin(), preds_at_step.end());
    }
}

/**
 * The main driver method that fits the Fourier decomposition based seasonal
 * model to all the seasonal head nodes.
 */
void AnalysisGraph::fit_seasonal_head_node_model_via_fourier_decomposition() {

    // TODO: Just for debugging delete
    this->set_indicator_means_and_standard_deviations();

    // Group seasonal head nodes according to their seasonality.
    unordered_map<int, vector<int>> period_to_head_nodes;

    for (int head_node_id: this->head_nodes) {
        Node& hn = (*this)[head_node_id];
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
        // The maximum number of components we are going to evaluate in search
        // of the best number of components to be used.
        // max_k < period / 2 (Nyquist theorem)
        int max_k = period / 2;

        // Generate the maximum number of sinusoidal frequencies needed for the
        // period. By the Nyquist theorem this number is floor(period / 2)
        // The actual number of frequencies used to model a concept could be
        // less than this, which is decided by computing the root mean squared
        // error of the predictions for each number of components
        vector<double> period_freqs =
                           this->generate_frequencies_for_period(period, max_k);

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
        //       I am doing this to make assemble_head_node_modeling_LDS()
        //       method more generalized so that I could reuse it to assemble
        //       the final complete LDS with all the nodes (seasonal head nodes
        //       with different periods and body nodes).
        //       At the moment, in the complete system, head nodes does not
        //       occupy a contiguous range of rows in the transition matrix.
        unordered_map<int, int> hn_to_mat_row;
        for (int v = 0; v < hn_ids.size(); v++) {
            hn_to_mat_row[hn_ids[v]] = 2 * v;

            // Again in the light of reusing the LDS assembly code to assemble
            // the complete LDS
            Node& hn = (*this)[hn_ids[v]];
            hn.fourier_freqs = period_freqs;
        }

        for (int components = 0; components <= max_k; components++) {
            for (auto hn_id: hn_ids) {
                // Again in the light of reusing the LDS assembly code to assemble
                // the complete LDS
                Node& hn = (*this)[hn_id];
                hn.n_components = components;
            }

            this->compute_fourier_coefficients_from_least_square_optimization(
                                            sinusoidals, components, hn_ids);

            // Assemble the LDS that generates the head nodes with this
            // period using this number of components.
            auto [A_concept_period_base, s0_concept_period] =
                    this->assemble_head_node_modeling_LDS(A_sin_max_k_base,
                                                          s0_sin_max_k,
                                                          components,
                                                          hn_to_mat_row);

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

            // In the assemble_head_node_modeling_LDS() method
            // Node object members: n_components and fourier_coefficients are
            // accessed.
            // When we assemble the final transition matrix with all the fitted
            // seasonal head nodes and all the body nodes, we have to use Node
            // object members: best_n_components and best_fourier_coefficients,
            // for the final system to utilize the fitted seasonal models.
            // Therefore, we reassign best_n_components and
            // best_fourier_coefficient to n_components and
            // fourier_coefficients members so that we could reuse the
            // assemble_head_node_modeling_LDS() method
            // seamlessly without any change or state checking at the time of
            // assembling the transition matrix for the final complete system.
            hn.n_components = hn.best_n_components;
            hn.fourier_coefficients = hn.best_fourier_coefficients;
            hn.best_fourier_coefficients.resize(0);  // To save some memory

            if (hn.best_n_components > max_n_components_for_period) {
                max_n_components_for_period = hn.best_n_components;
            }
        }

        // Remember all the sinusoidal frequencies needed to fit the best
        // Fourier decomposition based seasonal model each of the
        // seasonal head nodes with this period.
        fourier_frequency_set.insert(period_freqs.begin(),
                            period_freqs.begin() + max_n_components_for_period);
    }

    // TODO: Just for debugging. Move to transition matrix assembly pari in
    // sampling.cpp to include all the body nodes as well. For the moment testing
    // Assemble the final LDS with all the variables
    vector<double> all_freqs(fourier_frequency_set.begin(),
                             fourier_frequency_set.end());
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

    auto [A_concept_full_base, s0_concept_full] =
                    this->assemble_head_node_modeling_LDS(A_sin_all_base,
                                                          s0_sin_all,
                                                          all_freqs.size(),
                                                          hn_to_mat_row);

    cout << A_concept_full_base << "\n";

    this->predictions_to_csv(A_concept_full_base, s0_concept_full, 24);
}
