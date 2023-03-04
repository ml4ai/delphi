import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.linalg import expm  # Marix exponential
import seaborn as sns

# from functools import partial
# import timeit
import time

# from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 8})
plt.rcParams['text.usetex'] = True
sns.set_style("whitegrid")


class Periodic_Time_Series:
    # constructor
    def __init__(self, id, period, name=''):
        self.id = id
        self.name = name
        self.name = ''
        self.period = period
        self.fourier_freqs = []  # Fourier frequencies used to model this node
        self.fourier_coefficients = []
        self.n_components = 0    # Number of fourier frequencies used to model this node
        self.data = []
        self.time_steps = []
        self.data_train = []
        self.time_steps_train = []
        self.data_validate = []
        self.time_steps_validate = []
        self.data_test = []
        self.time_steps_test = []
        self.partitioned_data = [[] for _ in range(self.period)]
        self.partitioned_data_train = [[] for _ in range(self.period)]
        self.partitioned_data_validation = [[] for _ in range(self.period)]
        self.partitioned_data_test = [[] for _ in range(self.period)]
        self.bin_means = []
        self.bin_means_train = []
        self.bin_means_validation = []
        self.bin_means_test = []
        self.tot_observations = 0

    def partition_data(self, train_periods, validate_periods):
        train = train_periods * self.period
        train = train if train > 0 else len(self.data)
        validate = train + validate_periods * self.period
        self.data_train = self.data[:train]
        self.time_steps_train = self.time_steps[:train]
        self.data_validate = self.data[train: validate]
        self.time_steps_validate = self.time_steps[train: validate]
        self.data_test = self.data[validate:]
        self.time_steps_test = self.time_steps[validate:]

        # Bins are 0, 1, 2, ..., self.period - 1
        bin_assignments = self.time_steps % self.period

        # partition == 0 => training
        # partition == 1 => validating
        # partition == 2 => testing
        partitions = self.time_steps // train
        if validate > 0:
            partitions += self.time_steps // validate
        partition_train = 0
        partition_validation = 1
        partition_test = 2

        self.partitioned_data_train = [np.array(self.data[np.logical_and(bin_assignments == bin, partitions == partition_train)])
                                       for bin in range(self.period)]
        self.partitioned_data_validation = [np.array(self.data[np.logical_and(bin_assignments == bin, partitions == partition_validation)])
                                            for bin in range(self.period)]
        self.partitioned_data_test = [np.array(self.data[np.logical_and(bin_assignments == bin, partitions == partition_test)])
                                            for bin in range(self.period)]

        self.bin_means_train = np.mean(self.partitioned_data_train, axis=1)
        self.bin_means_validation = np.mean(self.partitioned_data_validation, axis=1)
        self.bin_means_test = np.mean(self.partitioned_data_test, axis=1)

        for idx, ts in enumerate(self.time_steps):
            partition = ts % self.period
            self.partitioned_data[partition].append(self.data[idx])

        for partition in range(self.period):
            self.partitioned_data[partition] = np.array(self.partitioned_data[partition])
            self.bin_means.append(np.mean(self.partitioned_data[partition]))

    def put_data(self, data, time_steps, train_periods=0, validate_periods=0):
        self.tot_observations = min(len(data), len(time_steps))

        if len(data) != len(time_steps):
            print(f'{self.name}: ERROR - The number of time steps must be equal to the number of data points!')
            print(f'\t\tUsing only {self.tot_observations} data points')

        self.data = np.array(data[: self.tot_observations])
        self.time_steps = np.array(time_steps[: self.tot_observations])

        self.partition_data(train_periods, validate_periods)

    def train_validate_test_split(self, train_periods, validate_periods):
        train = train_periods * self.period
        validate = train + validate_periods * self.period
        self.data_train = self.data[:train]
        self.time_steps_train = self.timesteps[:train]
        self.data_validate = self.data[train: validate]
        self.time_steps_validate = self.timesteps[train: validate]
        self.data_test = self.data[validate:]
        self.time_steps_test = self.timesteps[validate:]


'''
Generates a vector of all the effective frequencies for a particular period
@param n_components: The number of pure sinusoidal frequencies to generate
@param period: The period of the variable(s) being modeled by Fourier
               decomposition. All variable(s) share the same period.
@return A vector of all the effective frequencies
'''
def generate_frequencies_for_period(period, n_components):
    freqs = np.zeros(n_components)

    # λ is the amount we have to stretch/shrink pure sinusoidal curves to make
    # one sine cycle = period radians
    lmda = 2.0 * np.pi / period

    # ω is the frequency of each stretched/shrunk sinusoidal curve.
    # With respect to trigonometric function differentiation, the effective
    # frequency of pure sinusoidal curves is λω
    for omega in range(1, n_components + 1):
        freqs[omega - 1] = lmda * omega

    return freqs


'''
Assemble the LDS to generate sinusoidals of desired effective frequencies
@param freqs: A vector of effective frequencies
              (2πω / period; ω = 1, 2, ...)
@return pair (base transition matrix, initial state)
        0 radians is the initial angle.
'''
def assemble_sinusoidal_generating_LDS(freqs):
    lds_size = len(freqs) * 2

    # For each effective frequency, there are two rows in the transition matrix:
    #      1) for first derivative and
    #      2) for second derivative
    # In the state vector these rows are:
    #      1) the value
    #      2) the first derivative---
    A_sin = np.zeros((lds_size, lds_size))
    s0_sin = np.zeros(lds_size)

    for i in range(len(freqs)):
        i2 = i * 2     # first derivative matrix row
        i2p1 = i2 + 1  # second derivative matrix row

        # Assembling a block diagonal matrix
        # with each 2 x 2 block having the format
        # columns ->   2i        2i+1
        #            _________________
        # row 2i    |   0          1 |
        # row 2i+1  | -(λω)^2      0 |
        # ω = i+1 & λ = 2π/period
        A_sin[i2][i2p1] = 1
        A_sin[i2p1][i2] = -freqs[i] * freqs[i]

        # Considering t₀ = 0 radians, the initial
        # state of the sinusoidal generating LDS is:
        # row 2i    |    sin(λω 0) |  = |  0 |
        # row 2i+1  | λω cos(λω 0) |    | λω |
        # ω = i+1 & λ = 2π/period
        s0_sin[i2p1] = freqs[i]

    return A_sin, s0_sin


'''
Evolves the provided LDS (A_base and _s0) n_modeling_time_steps (e.g. months)
taking step_size steps each time. For example, if the step_size is 0.25,
there will be 4 prediction points per one modeling time step.
@param A_base: Base transition matrix that define the LDS.
@param _s0: Initial state of the system. Used _s0 instead of s0 because s0 is
            the member variable that represent the initial state of the final
            system that is used by the AnalysisGraph object.
@param n_modeling_time_steps: The number of modeling time steps (full time
                              steps, e.g. months) to evolve the system.
@param step_size: Amount to advance the system at each step.
@return A matrix of evolved values. Each column has values for one step.
             row 2i   - Values for variable i in the system
             row 2i+1 - Derivatives for variable i in the system
'''
def evolve_LDS(A_base, _s0, n_modeling_time_steps, step_size):
    tot_steps = int(n_modeling_time_steps / step_size)
    lds_size = len(_s0)

    # Transition matrix to advance the system one step_size forward
    A_step = expm(A_base * step_size)

    # A matrix to accumulate predictions of the system
    preds = np.zeros((lds_size, tot_steps))
    preds[:, 0] = _s0

    # Evolve the LDS one step_size at a time for desired number of steps
    for col in range(1, tot_steps):
        preds[:, col] = A_step @ preds[:, col - 1]

    return preds


'''
Generate a matrix of sinusoidal values of all the desired effective
frequencies for all the bin locations.
@param A_sin_base: Base transition matrix for sinusoidal generating LDS
@param s0_sin: Initial state (0 radians) for sinusoidal generating LDS
@param period: Period shared by the time series that will be fitted using
               the generated sinusoidal values. This period must be the
               same period used to generate the vector of effective
               frequencies used to assemble the transition matrix
               A_sin_base and the initial state s0_sin.
@return A matrix of required sinusoidal values.
        row t contains sinusoidals for bin t (radians)
        col 2ω contains sin(λω t)
        col 2ω+1 contains λω cos(λω t)
        with ω = 1, 2, ... & λ = 2π/period
'''
def generate_sinusoidal_values_for_bins(A_sin_base, s0_sin, period):
    # Evolve the sinusoidal generating LDS one step at a time for a whole
    # period to generate all the sinusoidal values required for the Fourier
    # reconstruction of the seasonal time series. Column t provides
    # sinusoidal values required for bin t.
    sinusoidals = evolve_LDS(A_sin_base, s0_sin, period, 1)

    # Transpose the sinusoidal matrix so that row t
    # contains the sinusoidal values for bin t.
    return sinusoidals.T


# NOTE: This method could be made a method of the Periodic_Time_Series class. The best
#       architecture would be to make a subclass, Periodic_Time_Series, of a class Time_Series and
#       include this method there.
'''
For each head node, computes the Fourier coefficients to fit a seasonal
curve to partitioned observations (bins) using the least square
optimization.
@param sinusoidals: Sinusoidal values of required effective frequencies at
                    each bin position. Row b contains all the sinusoidal
                    values for bin b.
                       sinusoidals(b, 2(ω-1))     =    sin(λω b)
                       sinusoidals(b, 2(ω-1) + 1) = λω cos(λω b)
                   with ω = 1, 2, ... & λ = 2π/period
@param n_components: The number of different sinusoidal frequencies used to
                     fit the seasonal head node model. The supplied
                     sinusoidals matrix could have sinusoidal values for
                     higher frequencies than needed
                     (number of columns > 2 * n_components). This method
                     utilizes only the first 2 * n_components columns.
@param head_node_ids: A list of head nodes with the period matching the
                      period represented in the provided sinusoidals. The
                      period of all the head nodes in this list must be the
                      same as the period parameter used when generating the
                      sinusoidals.
@return The Fourier coefficients in the order: α₀, β₁, α₁, β₂, α₂, ...
        α₀ is the coefficient for    cos(0)/2  term
        αᵢ is the coefficient for λi cos(λi b) term
        βᵢ is the coefficient for    sin(λi b) term
        with i = 1, 2, ... & λ = 2π/period & b = 0, 1, ..., period - 1
'''
def compute_fourier_coefficients_from_least_square_optimization(sinusoidals, n_components, head_nodes):
    tot_sinusoidal_rows = 2 * n_components
    components2p1 = tot_sinusoidal_rows + 1

    for hn in head_nodes:
        '''
        Setting up the linear system Ux = y to solve for the
        Fourier coefficients using the least squares optimization
        '''
        # Adding one additional column for cos(0) term
        U = np.zeros((hn.tot_observations, tot_sinusoidal_rows + 1))

        # Setting the coefficient for cos(0) term (α₀).
        # In the traditional Fourier decomposition this is 0.5
        # and when α₀ is used, we have to divide it by 2.
        # To avoid this additional division, here we use 1 instead.
        U[:, 0] = np.ones(hn.tot_observations)  # ≡ cos(0)

        y = np.zeros(hn.tot_observations)

        row = 0

        # Iterate through all the bins (partitions)
        for bin, data in enumerate(hn.partitioned_data):

            # Iterate through all the observations in one bin
            for obs in data:
                # Only the first tot_sinusoidal_rows
                # columns of the sinusoidals are used.
                U[np.ix_([row], np.arange(1, components2p1))] = sinusoidals[bin, 0:tot_sinusoidal_rows]
                y[row] = obs
                row += 1

        x, residuals, rank, s = np.linalg.lstsq(U, y, rcond=None)
        hn.fourier_coefficients = x


'''
Assembles the LDS to generate the head nodes specified in the hn_to_mat_row
map by Fourier reconstruction using the sinusoidal frequencies generated by
the provided sinusoidal generating LDS: A_sin_base and s0_sin.
@param A_sin_base: Base transition matrix to generate sinusoidals with all
                   the possible effective frequencies.
@param s0_sin: Initial state of the LDS that generates sinusoidal curves.
@param n_components: The number of different sinusoidal frequencies used to
                     fit the seasonal head nodes. The sinusoidal generating
                     LDS provided (A_sin_base, s0_sin) could generate more
                     sinusoidals of higher frequencies. This method uses
                     only the lowest n_components frequencies to assemble
                     the complete system.
@param n_concepts: The number of concepts being modeled by thi LDS. For the
                   LDS to correctly include all the seasonal head nodes
                   specified in hn_to_mat_row:
                   n_concepts ≥ hn_to_mat_row.size()
@param hn_to_mat_row: A map that maps each head node being modeled to the
                      transition matrix rows and state vector rows.
                      Each concept is allocated to two consecutive rows.
                      In the transition matrix:
                            even row) for first derivative
                            odd  row) for second derivative
                      In the state vector:
                            even row) the value
                            odd  row) for first derivative and
                      This map indicates the even row numbers each concept
                      is assigned to.
@return pair (base transition matrix, initial state) for the complete LDS
        with the specified number of sinusoidal frequencies (n_components).
        0 radians is the initial angle.
'''
def assemble_head_node_modeling_LDS(A_sin_base, s0_sin, n_components, n_concepts, hn_to_mat_row):
    tot_concept_rows = 2 * n_concepts
    tot_sinusoidal_rows = 2 * n_components
    lds_size = tot_concept_rows + tot_sinusoidal_rows

    frequency_to_idx = {}
    for i in range(0, tot_sinusoidal_rows, 2):
        # Since we are using t₀ = 0 radians as the initial time point, the
        # odd rows of the initial state of the sinusoidal generating LDS,
        # s0_sin, contains all the effective frequencies (λω) used to generate
        # the sinusoidal curves of different effective frequencies. Here we
        # are extracting those and assign them to the rows of the complete LDS.
        # The first tot_concept_rows of the system are used to model the actual
        # concepts. The latter tot_sinusoidal_rows are used to generate the
        # sinusoidal curves of all the effective frequencies used to model the
        # seasonal head nodes.
        frequency_to_idx[s0_sin[i + 1]] = tot_concept_rows + i

    A_complete_base = np.zeros((lds_size, lds_size))

    A_complete_base[-tot_sinusoidal_rows :, -tot_sinusoidal_rows :] = \
        A_sin_base[: tot_sinusoidal_rows, : tot_sinusoidal_rows]

    s0_complete = np.zeros(lds_size)
    s0_complete[-tot_sinusoidal_rows :] = s0_sin[: tot_sinusoidal_rows]

    for hn, dot_row in hn_to_mat_row.items():
        dot_dot_row = dot_row + 1

        # Sinusoidal coefficient vector to calculate initial value for concept v
        v0 = np.zeros(len(hn.fourier_coefficients))

        # Coefficient for cos(0) term (α₀). In the traditional Fourier
        # decomposition this is 0.5. When we compute α₀ we include this factor
        # into α₀ (Instead of computing α₀ as in the traditional Fourier series,
        # we compute α₀/2 straightaway).
        v0[0] = 1

        for concept_freq_idx in range(0, hn.n_components):

            concept_freq = hn.fourier_freqs[concept_freq_idx]  # λω
            concept_freq_squared = concept_freq * concept_freq

            concept_freq_idx_2 = concept_freq_idx * 2
            beta_omega_idx = concept_freq_idx_2 + 1
            alpha_omega_idx = concept_freq_idx_2 + 2

            # Coefficient for sin(λω t) terms ≡ β_ω
            beta_omega = hn.fourier_coefficients[beta_omega_idx]
            # Coefficient for λω cos(λω t) terms ≡ α_ω
            alpha_omega = hn.fourier_coefficients[alpha_omega_idx]

            sin_idx = frequency_to_idx[concept_freq]
            cos_idx = sin_idx + 1

            # Setting coefficients of the first derivative of the head node
            # hn_id. They are in row 2 * hn_id in the transition matrix.

            # Setting coefficients for sin terms: -freq^2 * cos_coefficient
            A_complete_base[dot_row, sin_idx] = -concept_freq_squared * alpha_omega

            # Setting coefficients for cos terms: sin_coefficient
            A_complete_base[dot_row, cos_idx] = beta_omega

            # Setting coefficients of the second derivative of the head node
            # hn_id. They are in row 2 * hn_id + 1 in the transition matrix.

            # Setting coefficients for sin terms: -freq^2 * sin_coefficient
            A_complete_base[dot_dot_row, sin_idx] = -concept_freq_squared * beta_omega

            # Setting coefficients for cos terms: -freq^2 * cos_coefficient
            A_complete_base[dot_dot_row, cos_idx] = -concept_freq_squared * alpha_omega

            # Populating the sinusoidal coefficient vector to compute the
            # initial value for the head node hn_id.
            v0[beta_omega_idx] = s0_complete[sin_idx]
            v0[alpha_omega_idx] = s0_complete[cos_idx]

        # Setting the initial value for head node hn_id.
        s0_complete[dot_row] = hn.fourier_coefficients.dot(v0)

        # Setting the initial derivative for head node hn_id.
        s0_complete[dot_dot_row] = A_complete_base[dot_row, :].dot(s0_complete)

    return A_complete_base, s0_complete


'''
Split data by contiguous blocks of full periods
'''
class Custom_Cross_Validation_period_blocks:
    @classmethod
    def split(cls, X, period=12):
        n_time_steps = X.shape[0]
        periods = n_time_steps // period
        n_folds = min(10, periods)

        n_time_steps_pre_fold = n_time_steps // n_folds
        n_full_periods_per_fold = n_time_steps_pre_fold // period
        n_full_period_timesteps_per_fold = n_full_periods_per_fold * period

        validation_fold_start = 0

        for fold in range(n_folds):
            validation_fold_end = validation_fold_start + n_full_period_timesteps_per_fold

            validation_fold = np.arange(validation_fold_start, validation_fold_end, dtype=int)
            train_folds = np.concatenate((np.arange(0, validation_fold_start, dtype=int),
                                          np.arange(validation_fold_end, n_time_steps, dtype=int)))

            yield train_folds, validation_fold

            validation_fold_start = validation_fold_end


def plot_cv_regression_results(model):
    lasso = model
    plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
    plt.plot(
        lasso.alphas_,
        lasso.mse_path_.mean(axis=-1),
        color="black",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

    plt.xlabel(r"$\alpha$")
    plt.ylabel("Mean square error")
    plt.legend()
    _ = plt.title(
        f"Mean square error on each fold: coordinate descent (train time: {0:.2f}s)"
    )
    plt.show()


def regression(pts):
    # The maximum number of components we are going to evaluate
    # in search of the best number of components to be used.
    # max_k < period / 2 (Nyquist theorem)
    max_k = 10 #hn.period // 2;

    # Generate the maximum number of sinusoidal frequencies needed for the
    # period. By the Nyquist theorem this number is floor(period / 2)
    # The actual number of frequencies used to model a concept could be
    # less than this, which is decided by computing the root mean squared
    # error of the predictions for each number of components
    period_freqs = generate_frequencies_for_period(pts.period, max_k)

    pred_step = 0.1
    pred_periods = 3
    pred_time_steps = np.arange(0, pts.period * pred_periods, pred_step)
    sinusoidals = np.zeros((len(period_freqs) * 2, len(pts.time_steps_train)))
    pred_sinusoidals = np.zeros((len(period_freqs) * 2, len(pred_time_steps)))

    for idx, freq in enumerate(period_freqs):
        sinusoidals[idx * 2] = np.sin(pts.time_steps_train * freq)
        sinusoidals[idx * 2 + 1] = np.cos(pts.time_steps_train * freq)

        pred_sinusoidals[idx * 2] = np.sin(pred_time_steps * freq)
        pred_sinusoidals[idx * 2 + 1] = np.cos(pred_time_steps * freq)

    lr_cv_models = {'LassoCV': LassoCV, 'RidgeCV': RidgeCV, 'ElasticNetCV': ElasticNetCV}
    lr_models = ['LR'] + list(lr_cv_models.keys())

    scores = {'k': np.arange(1, max_k + 1, dtype=int)}
    scores.update({
        f'{model} {criteria}': np.zeros(max_k, dtype=float)
        for criteria in ['RMSE', 'R^2']
        for model in lr_models
    })

    params = [ab + str(freq)
              for freq in range(1, max_k + 1)
              for ab in ['A', 'B']]
    params.insert(0, 'A0')
    params.append('Alpha')

    parameters = {'Parameter': params}
    parameters.update({
        f'{model} ({freq})': np.zeros(2 * max_k + 2, dtype=float)
        for freq in range(1, max_k + 1)
        for model in lr_models
    })

    predictions = {}

    for n_components in range(1, max_k + 1):
        n_sinu_rows = 2 * n_components
        summary_row = n_components - 1

        lr = LinearRegression()
        lr.fit(sinusoidals[:n_sinu_rows, :].T, pts.data_train)

        pred_train_lr = lr.predict(pred_sinusoidals[:n_sinu_rows, :].T)
        pred_test = lr.predict(sinusoidals[:n_sinu_rows, :len(pts.data_test)].T)

        predictions['LR'] = pred_train_lr

        scores['LR RMSE'][summary_row] = sqrt(mean_squared_error(pts.data_test, pred_test))
        scores['LR R^2'][summary_row] = lr.score(sinusoidals[:n_sinu_rows, :len(pts.data_test)].T, pts.data_test)
        parameters[f'LR ({n_components})'][0] = lr.intercept_
        parameters[f'LR ({n_components})'][1:n_sinu_rows + 1] = lr.coef_
        parameters[f'LR ({n_components})'][-1] = 0

        if n_components == 6:
            largest_coef = np.argmax(np.abs(lr.coef_))
            # I get: 1.32365885e+13
            print(f'\nCoefficients of the LR Model with {n_components} components:\n\n'
                  f'Notice the coefficient with the largest absolute magnitude at index {largest_coef}:\n\n'
                  f'\t\t\t{lr.coef_[largest_coef]}\n\n', lr.coef_)

        for model_name, LR_CV in lr_cv_models.items():
            custom_cv_block_periods_12 = Custom_Cross_Validation_period_blocks.split(sinusoidals[:n_sinu_rows, :].T, pts.period)
            lr_cv = LR_CV(cv=custom_cv_block_periods_12).fit(sinusoidals[:n_sinu_rows, :].T, pts.data_train)

            pred_train_lr_cv= lr_cv.predict(pred_sinusoidals[:n_sinu_rows, :].T)
            pred_test = lr_cv.predict(sinusoidals[:n_sinu_rows, :len(pts.data_test)].T)

            predictions[model_name] = pred_train_lr_cv

            scores[f'{model_name} RMSE'][summary_row] = sqrt(mean_squared_error(pts.data_test, pred_test))
            scores[f'{model_name} R^2'][summary_row] = lr.score(sinusoidals[:n_sinu_rows, :len(pts.data_test)].T, pts.data_test)
            parameters[f'{model_name} ({n_components})'][0] = lr_cv.intercept_
            parameters[f'{model_name} ({n_components})'][1:n_sinu_rows + 1] = lr_cv.coef_
            parameters[f'{model_name} ({n_components})'][-1] = lr_cv.alpha_

        '''Plotting'''

        # With unregularized least squares (LinearRegression), the curve only fits well at the
        # observation locations. At intermediate points, the curve fluctuates wildly between
        # extremely high values at the Nyquist frequency and above (E.g., for a period = 12
        # time series, using frequencies 6 or above)
        # Setting obs_locations = 1 will show this behavior in the plots
        obs_locations = int(1 / pred_step)
        sns.lineplot(x=pred_time_steps[::obs_locations], y=predictions['LR'][::obs_locations], label='LR')

        for model_name in lr_cv_models.keys():
            sns.lineplot(x=pred_time_steps, y=predictions[model_name], label=model_name)

        plt.title(f'Fourier Frequencies: {n_components}')
        plt.legend()
        plt.show()
        plt.close()

    '''Saving results to csv files'''

    rmse_plot = {'Models': []}
    rmse_plot.update({k: [] for k in np.arange(1, max_k + 1)})

    for model_name in lr_cv_models.keys():
        rmse_plot['Models'].append(model_name)
        for n_components in scores['k']:
            rmse_plot[n_components].append(scores[f'{model_name} RMSE'][n_components - 1])

    df_scores = pd.DataFrame(scores)
    df_rmse_plot = pd.DataFrame(rmse_plot)
    df_parameters = pd.DataFrame(parameters)

    df_scores.to_csv('linear_regression_scores.csv', index=False)
    df_rmse_plot.to_csv('linear_regression_rmse.csv', index=False)
    df_parameters.to_csv('linear_regression_parameters.csv', index=False)

'''
Testing linear regression
'''
# df_rain = pd.read_csv('../../uai_2023/data/synthetic/fourier/6_Fourier_period=12_pure-std=6.142_noise-std=1.228_noise-factor=5.csv')
# rain = np.array(df_rain['Noisy'].tolist())
# df_rain = pd.read_csv('../../uai_2023/data/real/TerraClimate_Oromia_Monthly_Precip.csv')
# rain = np.array(df_rain['Precipitation'].tolist())
# months = np.array(range(len(rain)))
#
# period = 12
# name = 'Precipitation'
# pts_rain = Periodic_Time_Series(0, period, name)
# pts_rain.put_data(rain, months, 32, 20)
#
# regression(pts_rain)
# exit()


def extract_coeff_exponent10(number) -> (float, int):
    expo = int(np.floor(np.log10(np.abs(number))))
    return number * (10 ** np.abs(expo)), expo


def benchmark_matrix_exponential():
    name = 'Accent'
    cmap = get_cmap('tab10')
    colors = cmap.colors

    repetitions = 5
    eta_times = []
    mat_sizes = []
    min_matrix_size = 2
    max_matrix_size = 1000
    step_size = 10

    for repeat in range(repetitions):
        print(repeat)
        for period in range(min_matrix_size, max_matrix_size + step_size, step_size):
            # print(repeat, period)
            max_k = period // 2
            period_freqs = generate_frequencies_for_period(period, max_k)
            A_sin_max_k_base, s0_sin_max_k = assemble_sinusoidal_generating_LDS(period_freqs)
            # A_sin_max_k_base = np.random.rand(period, period)
            start_time = time.perf_counter()
            eta = expm(A_sin_max_k_base * step_size)
            eta_time = time.perf_counter() - start_time
            eta_times.append(eta_time)
            mat_sizes.append(period)

    # df_me_size_vs_time =pd.read_csv('me_size_vs_time_1000_30.csv')
    df_me_size_vs_time = pd.DataFrame({'Matrix size': mat_sizes, 'Runtime (ms)': np.array(eta_times) / 1000})
    df_me_size_vs_time.to_csv(f'me_size_vs_time_{max_matrix_size}_{repeat}.csv', index=False)

    df_size_group = df_me_size_vs_time.groupby(by='Matrix size').describe()
    df_size_group.to_csv(f'me_size_vs_time_{max_matrix_size}_{repeat}_stats.csv')

    # fig, ax = plt.subplots(dpi=250, figsize=(3.25, 2))
    fig, ax = plt.subplots(dpi=250, figsize=(15, 7))
    ax.set_prop_cycle(color=colors)
    sns.lineplot(data=df_me_size_vs_time, x='Matrix size', y='Runtime (ms)', linewidth=0.5, label='Runtimes')
    # plt.scatter(x=df_me_size_vs_time['Matrix size'], y=df_me_size_vs_time['Runtime (ms)'], c='red', s=0.01, alpha=0.5)

    # calculate equation for quadratic trendline
    truncate = 0
    df_truncated = df_me_size_vs_time[df_me_size_vs_time['Matrix size'] > min_matrix_size + truncate]
    z = np.polyfit(x=df_truncated['Matrix size'], y=df_truncated['Runtime (ms)'], deg=3)
    # z = np.polyfit(x=df_me_size_vs_time['Matrix size'], y=df_me_size_vs_time['Runtime (ms)'], deg=3)
    # z = np.polynomial.polynomial.Polynomial.fit(x=df_me_size_vs_time['Matrix size'], y=df_me_size_vs_time['Runtime (ms)'], deg=3)
    # print(z)
    # print(z.convert().coef)
    p = np.poly1d(z)
    # p = np.polynomial.polynomial.Polynomial(z.convert().coef)
    print('Squared error: ', np.sum((p(df_truncated['Matrix size']) - df_truncated['Runtime (ms)'])**2))
    # 3 = 2.471400456865427e-06
    # 2 = 2.792930890674089e-06
    # 1 = 8.778211571902997e-06
    # exit()

    coeff_eponents = []
    for coeff in z:
        coeff_eponents.append(extract_coeff_exponent10(coeff))

    # add trendline to plot
    lbl = f'$y=10^{{-7}}\\times \\big({coeff_eponents[0][0]:0.3}\\times 10^{{{coeff_eponents[0][1] + 7}}}\,x^3 + ' \
          f'{coeff_eponents[1][0]:0.3}\\times 10^{{{coeff_eponents[1][1] + 7}}}\,x^2 + ' \
          f'{coeff_eponents[2][0]:0.3}\,x + ' \
          f'{coeff_eponents[3][0] * 10:0.3}\\big)$'
          # f'{coeff_eponents[2][0]:0.3}\\times 10^{{{coeff_eponents[2][1] + 6}}}\,x + ' \
          # f'{coeff_eponents[3][0]:0.3}\\times 10^{{{coeff_eponents[3][1] + 6}}}\\big)$'
    x = list(range(min_matrix_size + truncate, max_matrix_size + step_size, 2))
    sns.lineplot(x=x, y=p(x), linewidth=0.75, label=lbl)

    # plt.title(f'$y={z[0]:0.3}x^3 + {z[1]:0.3}x^2 + {z[2]:0.3}x + {z[3]:0.3}$')
    # plt.title(f'$y={coeff_eponents[0][0]:0.3}\\times 10^{{{coeff_eponents[0][1]}}}\,x^3 + {z[1]:0.3}x^2 + {z[2]:0.3}x + {z[3]:0.3}$')
    # plt.title(f'$y={coeff_eponents[0][0]:0.3}\\times 10^{{{coeff_eponents[0][1]}}}\,x^3 + {coeff_eponents[1][0]:0.3}\\times 10^{{{coeff_eponents[1][1]}}}\,x^2 + {coeff_eponents[2][0]:0.3}\\times 10^{{{coeff_eponents[2][1]}}}\,x + {coeff_eponents[3][0]:0.3}\\times 10^{{{coeff_eponents[3][1]}}}\,$')
    # plt.title(f'$y={coeff_eponents[0][0]:0.3}\\times 10^{{{coeff_eponents[0][1]}}}\,x^2 + {coeff_eponents[1][0]:0.3}\\times 10^{{{coeff_eponents[1][1]}}}\,x + {coeff_eponents[2][0]:0.3}\\times 10^{{{coeff_eponents[2][1]}}}$')
    plt.xlabel('Matrix size')
    plt.ylabel('Runtime (ms)')
    plt.xlim(0, max_matrix_size)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    # plt.savefig(f'me_size_vs_time_{max_matrix_size}_{repeat}.pdf')
    plt.savefig('me_size_vs_time.pdf')
    plt.close()


def benchmark_matrix_exponential_cashing():
    name = 'Accent'
    cmap = get_cmap('tab10')
    colors = cmap.colors

    repititions = 40
    n_timesteps = 100
    n_unique_gaps = 1
    possible_gaps = np.arange(start=1, stop=n_timesteps + 1)

    periods = {4: 'r', 8: 'g', 16: 'b'}

    nugs = {f'Num Unique Gaps ({period})': [] for period in periods}
    me_efficient = {f'Runtime Gaps ({period}) (ms)': [] for period in periods}
    me_conventional = {f'Runtime Time (ms) ({period}) (ms)': [] for period in periods}

    # period = 4
    for period in periods.keys():
        max_k = period // 2
        period_freqs = generate_frequencies_for_period(period, max_k)
        A_sin_max_k_base, s0_sin_max_k = assemble_sinusoidal_generating_LDS(period_freqs)

        for repeat in range(repititions):
            print(repeat)
            for n_unique_gaps in range(1, n_timesteps + 1):
                # When n_unique_gaps == 1, time steps are evenly sampled, and we only need to compute one e^tA
                # this provides the maximum benefit of cashing.
                # When n_unique_gaps == n_timesteps, each time step is at a distinct distance from the previous
                # time step (no two time gaps are the same). Therefore, we must compute e^tA n_timesteps times.
                # Caching does not give any benefit in this case.

                # Select n_unique_gaps gaps from {1, 2, 3, ..., n_timesteps}
                gaps = np.random.choice(possible_gaps, n_unique_gaps, replace=False)

                # Decide how many times each selected gap is repeated to get n_timesteps time steps.
                # Resample n_timesteps - n_unique_gaps with replacement from selected gaps
                gaps = np.concatenate((gaps, np.random.choice(gaps, n_timesteps - n_unique_gaps, replace=True)))

                # Cumulatively add the gaps to get the time step sequence
                # Gaps:         3, 2, 4,  1,  2,  3, ...
                # Time steps:   3, 5, 9, 10, 12, 15, ...
                timsteps = np.cumsum(gaps)

                # Due to the algorithm for selecting the gaps, the number of actual_unique_gaps
                # should be equal to n_unique_gaps
                actual_unique_gaps = set(gaps)
                # print(n_unique_gaps, len(actual_unique_gaps))
                nugs[f'Num Unique Gaps ({period})'].append(len(actual_unique_gaps))

                me_for_gaps = {}
                start_time = time.perf_counter()

                # Compute e^tA actual_unique_gaps times and memorize
                for unique_gap in actual_unique_gaps:
                    me_for_gaps[unique_gap] = expm(A_sin_max_k_base * unique_gap)

                # Access the memorized e^tAs as per the gap sequence for n_timesteps times
                for gap in gaps:
                    eta = me_for_gaps[gap]
                eta_time = time.perf_counter() - start_time
                me_efficient[f'Runtime Gaps ({period}) (ms)'].append(eta_time / 1000)

                start_time = time.perf_counter()

                # Compute e^tA n_timesteps times based on the time step sequence
                for ts in timsteps:
                    eta = expm(A_sin_max_k_base * ts)
                eta_time = time.perf_counter() - start_time
                me_conventional[f'Runtime Time (ms) ({period}) (ms)'].append(eta_time / 1000)

    fig, ax = plt.subplots(dpi=250, figsize=(3.25, 3.7))
    ax.set_prop_cycle(color=colors)

    for period, color in periods.items():
        sns.lineplot(x=nugs[f'Num Unique Gaps ({period})'], y=me_efficient[f'Runtime Gaps ({period}) (ms)'],
                     color=color, linewidth=0.75, label=f'Economic ({period})')
        sns.lineplot(x=nugs[f'Num Unique Gaps ({period})'], y=me_conventional[f'Runtime Time (ms) ({period}) (ms)'],
                     color=color, linewidth=0.75, label=f'Conventional ({period})', linestyle='dashed')

    # Prepare a custom legend
    legend_elements = [Line2D([0], [0], color='w', lw=0.75, label='Matrix Size')]
    for period, color in reversed(periods.items()):
        legend_elements.append(Line2D([0], [0], color=color, lw=0.75, label=f'{period}'))

    legend_elements += [Line2D([0], [0], color='w', lw=0.75, label=''),
                        Line2D([0], [0], color='k', lw=0.75, label='Conventional', linestyle='dashed'),
                        Line2D([0], [0], color='k', lw=0.75, label='Economic')]

    ax.legend(handles=legend_elements, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.15))

    plt.xlabel('Number of Unique Gaps')
    plt.ylabel('Runtime (ms)')

    # plt.xlim(0, n_timesteps)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'me_economic_{min(periods.keys())}-{max(periods.keys())}.pdf')
    plt.close()

    nugs.update(me_efficient)
    nugs.update(me_conventional)
    df_me_econo_time = pd.DataFrame(nugs)
    df_me_econo_time.to_csv(f'me_economic_{min(periods.keys())}-{max(periods.keys())}.csv', index=False)

    df_unique_gaps_group = df_me_econo_time.groupby(by=f'Num Unique Gaps ({min(periods.keys())})').describe()
    df_unique_gaps_group.to_csv(f'me_economic_{min(periods.keys())}-{max(periods.keys())}_stats.csv')


def benchmark_matrix_exponential_accuracy():
    name = 'Accent'
    cmap = get_cmap('tab10')
    colors = cmap.colors

    period = 4
    max_k = period // 2
    period_freqs = generate_frequencies_for_period(period, max_k)
    A_sin_max_k_base, s0_sin_max_k = assemble_sinusoidal_generating_LDS(period_freqs)

    # Make t_0 = pi / 2
    s0_sin_max_k = expm(A_sin_max_k_base) @ s0_sin_max_k

    # sin[½π(4n + 1)] = sin(½π) = 1, n = 0, 1, 2, ...
    # since we are making t_0 = pi / 2, we do not have to advance the system from 0 radians to ½π radians.
    # Therefore, we do not have to consider +1 in (4n + 1). This makes math easier within the loop
    n = 16
    final_ts = 4 * n
    true_value = 1
    step_size = final_ts
    n_steps = 1
    error = []
    step_sizes = []
    steps = []

    # Compute sin[½π(4n + 1)] multiple times using different step sizes (and number of steps) with the initial state
    # sin(½π)
    # final_ts (1), final_ts/2 (2), final_ts/4 (4), final_ts/8 (8), ....
    while step_size > 0.001:
        step_size = final_ts / n_steps
        print(step_size, n_steps, n_steps * step_size)
        s = s0_sin_max_k
        for _ in range(n_steps):
            s = expm(A_sin_max_k_base * step_size) @ s
        step_sizes.append(step_size)
        steps.append(n_steps)
        error.append(s[0] - true_value)
        n_steps *= 2

    df_me_accuracy = pd.DataFrame({'Step size': step_sizes,
                                   'Number of Steps': steps,
                                   'Error': error})
    df_me_accuracy.to_csv(f'me_accuracy.csv', index=False)

    print(error)
    print(step_sizes)
    zoom_min = 3
    zoom_max = 9
    fig, ax = plt.subplots(dpi=250, figsize=(3.25, 2.5))
    ax.set_prop_cycle(color=colors)
    sns.lineplot(x=step_sizes, y=error, marker='o')
    # Mark the points shown in the zoomed in plot
    plt.scatter(x=step_sizes[zoom_min:zoom_max], y=error[zoom_min:zoom_max], c='red', s=5, zorder=10)
    plt.xlabel('$\log_2$(Step Size) ($\log_2(\delta t)$)')
    plt.ylabel('$error = \\big(e^{\delta t\,A}\\big)^\\frac{n}{\delta t}\,s_0 - 1$')

    plt.xscale('log', base=2)
    plt.tight_layout()

    # plot the zoomed portion
    # https://stackoverflow.com/questions/13583153/how-to-zoomed-a-portion-of-image-and-insert-in-the-same-plot-in-matplotlib
    # location for the zoomed portion
    sub_axes = plt.axes([.45, .57, .5, .37])

    sns.lineplot(x=step_sizes[zoom_min:zoom_max], y=error[zoom_min:zoom_max], marker='o', ax=sub_axes)
    plt.scatter(x=step_sizes[zoom_min:zoom_max], y=error[zoom_min:zoom_max], c='red', s=5, zorder=10)
    plt.xscale('log', base=2)

    # plt.show()
    plt.savefig(f'me_accuracy.pdf')
    plt.close()


# benchmark_matrix_exponential()
# benchmark_matrix_exponential_cashing()
# benchmark_matrix_exponential_accuracy()
# exit()

pred_step = 0.1
pred_periods = 7
slope = 0.5


def add_linear_trend(A_base, s0_period, slope):
    '''
    We add two more rows and columns to the transition matrix and two more rows to the state vector.
    The last two rows of the transition matrix and the state vector generate y = x curve.
    A = | 0 m |     and s_0 = | 0 |
        | 0 0 |               | 1 |
    y = e^tA @ s_0 generates line
    y = mt
    :param A_base: Fourier periodic model base transition matrix.
    :param s0_period: Fourier periodic model initial state.
    :param slope: Slope of the trend.
    :param intercept:
    :return: Base transition matrix and initial state extended with y = x line added, and the linear trend added to
             the periodic time series being modeled in the system.
    '''

    # Increase the size of the system by adding one more variable and its derivative to model the y = mx + c line
    # The last two rows of the transition matrix and the initial state models this
    lds_size = len(s0_period)
    lds_size_trend = lds_size + 2
    A_concept_period_base_trend = np.zeros((lds_size_trend, lds_size_trend))
    A_concept_period_base_trend[: lds_size, : lds_size] = A_base

    s0_concept_period_trend = np.zeros(lds_size_trend)
    s0_concept_period_trend[: lds_size] = s0_period

    # Generate the y = mx + c line within the system
    # Here we generate y = x line since the intercept (c) of the line does not affect the periodic time series and
    # with m = 1, it is easier to scale it to the desired linear slope of the periodic time series by m * slope
    m = 1                                  # slope of the line
    c = 0                                  # intercept of the line
    A_concept_period_base_trend[-2][-1] = 1
    s0_concept_period_trend[-1] = m
    s0_concept_period_trend[-2] = c

    # Scale the slope of the y = mx + c line by the desired slope, m * slope, and add to the curve at each step
    A_concept_period_base_trend[0][-1] = slope
    # A_concept_period_base_trend[0][-2] = 1

    return A_concept_period_base_trend, s0_concept_period_trend


def add_exponential_trend(A_base, s0_period, a):
    '''
    We add two more rows and columns to the transition matrix and two more rows to the state vector.
    The last two rows of the transition matrix and the state vector generate y = e^ta curve.
    A = | a  0 |     and s_0 = | 1 |
        | a² 0 |               | a |
    y = e^tA @ s_0 generates exponential
    y = e^ta
    :param A_base: Fourier periodic model base transition matrix.
    :param s0_period: Fourier periodic model initial state.
    :param a: Exponent.
    :return: Base transition matrix and initial state extended with y = e^ta curve added, and the exponential trend
             added to the periodic time series being modeled in the system.
    '''

    # Increase the size of the system by adding one more variable and its derivative to model the y = mx + c line
    # The last two rows of the transition matrix and the initial state models this
    lds_size = len(s0_period)
    lds_size_trend = lds_size + 2
    A_concept_period_base_trend = np.zeros((lds_size_trend, lds_size_trend))
    A_concept_period_base_trend[: lds_size, : lds_size] = A_base

    s0_concept_period_trend = np.zeros(lds_size_trend)
    s0_concept_period_trend[: lds_size] = s0_period

    # Generate the y = e^ta curve within the system
    A_concept_period_base_trend[-2, -2] = a
    A_concept_period_base_trend[-1, -2] = a * a
    s0_concept_period_trend[-2] = 1
    s0_concept_period_trend[-1] = a

    # Add the exponential trend to the periodic curve
    A_concept_period_base_trend[0][-1] = 1

    # A_concept_period_base_trend = np.zeros((2, 2))
    # s0_concept_period_trend = np.zeros(2)
    # A_concept_period_base_trend[-2, -2] = a
    # A_concept_period_base_trend[-1, -2] = a * a
    # s0_concept_period_trend[-2] = 1
    # s0_concept_period_trend[-1] = a

    return A_concept_period_base_trend, s0_concept_period_trend


def add_quadratic_trend(A_base, s0_period, a):
    '''
    We add two more rows and columns to the transition matrix and two more rows to the state vector.
    The last two rows of the transition matrix and the state vector generate y = x curve.
    A = | 0 m |     and s_0 = | 0 |
        | 0 0 |               | 1 |
    y = e^tA @ s_0 generates line
    y = mt
    :param A_base: Fourier periodic model base transition matrix.
    :param s0_period: Fourier periodic model initial state.
    :param a: Coefficient of the quadratic term.
    :return: Base transition matrix and initial state extended with y = x line added, and the linear trend added to
             the periodic time series being modeled in the system.
    '''
    '''
    The system
    A = | 0 0 2a 0 |     and s_0 = | 0  |   z
        | 0 0 0  0 |               | 2a |   z_dot
        | 0 0 0  1 |               | c  |   y
        | 0 0 0  0 |               | m  |   y_dot
    s_t = e^tA @ s_0 generates
    y = mt + c line as the third row of the state.
    When m = 1 and c = 0, this becomes y = t (time) line.
    Then the first row generates
    z = ay² = at²
    '''

    # Increase the size of the system by adding one more variable and its derivative to model the y = mx + c line
    # The last two rows of the transition matrix and the initial state models this
    lds_size = len(s0_period)
    lds_size_trend = lds_size + 2
    A_concept_period_base_trend = np.zeros((lds_size_trend, lds_size_trend))
    A_concept_period_base_trend[: lds_size, : lds_size] = A_base

    s0_concept_period_trend = np.zeros(lds_size_trend)
    s0_concept_period_trend[: lds_size] = s0_period

    # Generate the y = mx + c line within the system
    # Here we generate y = x line since the intercept (c) of the line does not affect the periodic time series and
    # with m = 1, it is easier to scale it to the desired linear slope of the periodic time series by m * slope
    m = 1                                  # slope of the line
    c = 0                                  # intercept of the line
    A_concept_period_base_trend[-2][-1] = 1
    s0_concept_period_trend[-1] = m
    s0_concept_period_trend[-2] = c

    # Add quadratic trend, at² to the periodic time series
    A_concept_period_base_trend[0, -2] = 2 * a

    # A_concept_period_base_trend = np.zeros((4, 4))
    # s0_concept_period_trend = np.zeros(4)
    #
    # # Generate the y = mx + c line within the system
    # # Here we generate y = x line since the intercept (c) of the line does not affect the periodic time series and
    # # with m = 1, it is easier to scale it to the desired linear slope of the periodic time series by m * slope
    # m = 1                                  # slope of the line
    # c = 0                                  # intercept of the line
    # A_concept_period_base_trend[-2][-1] = 1
    # s0_concept_period_trend[-1] = m
    # s0_concept_period_trend[-2] = c
    #
    # A_concept_period_base_trend[0, -2] = 2 * a
    # s0_concept_period_trend[1] = 2 * a
    #
    # print(A_concept_period_base_trend, '\n')
    # print(s0_concept_period_trend)

    return A_concept_period_base_trend, s0_concept_period_trend


def add_quadratic_and_linear_trend(A_base, s0_period, alpha, beta):
    '''
    The system
    A = | 0 0 2β α |     and s_0 = | 0  |   z
        | 0 0 0  1 |               | 2β |   z_dot
        | 0 0 0  1 |               | c  |   y
        | 0 0 0  0 |               | m  |   y_dot
    s_t = e^tA @ s_0 generates
    y = mt + c line as the third row of the state.
    When m = 1 and c = 0, this becomes y = t (time) line.
    Then the first row generates
    z = βy² + αy = βt² + αt
    '''

    # Increase the size of the system by adding one more variable and its derivative to model the y = mx + c line
    # The last two rows of the transition matrix and the initial state models this
    lds_size = len(s0_period)
    lds_size_trend = lds_size + 2
    A_concept_period_base_trend = np.zeros((lds_size_trend, lds_size_trend))
    A_concept_period_base_trend[: lds_size, : lds_size] = A_base

    s0_concept_period_trend = np.zeros(lds_size_trend)
    s0_concept_period_trend[: lds_size] = s0_period

    # Generate the y = mx + c line within the system
    # Here we generate y = x line since the intercept (c) of the line does not affect the periodic time series and
    # with m = 1, it is easier to scale it to the desired linear slope of the periodic time series by m * slope
    m = 1                                  # slope of the line
    c = 0                                  # intercept of the line
    A_concept_period_base_trend[-2][-1] = 1
    s0_concept_period_trend[-1] = m
    s0_concept_period_trend[-2] = c

    # Add quadratic trend, at² to the periodic time series
    A_concept_period_base_trend[0, -2] = 2 * beta
    A_concept_period_base_trend[0, -1] = alpha
    s0_concept_period_trend[1] = 2 * beta

    # A_concept_period_base_trend = np.zeros((4, 4))
    # s0_concept_period_trend = np.zeros(4)
    #
    # # Generate the y = mx + c line within the system
    # # Here we generate y = x line since the intercept (c) of the line does not affect the periodic time series and
    # # with m = 1, it is easier to scale it to the desired linear slope of the periodic time series by m * slope
    # m = 1                                  # slope of the line
    # c = 0                                  # intercept of the line
    # A_concept_period_base_trend[-2][-1] = 1
    # s0_concept_period_trend[-1] = m
    # s0_concept_period_trend[-2] = c
    #
    # A_concept_period_base_trend[0, -2] = 2 * beta
    # A_concept_period_base_trend[0, -1] = alpha
    # s0_concept_period_trend[1] = 2 * beta
    #
    # print(A_concept_period_base_trend, '\n')
    # print(s0_concept_period_trend)

    return A_concept_period_base_trend, s0_concept_period_trend


def add_sinusoidal_trend(A_base, s0_period, alpha, beta, period, frequency):
    '''
    The system with ω = frequency (< 1) and λ = 2π / period
    A = | 0 0    2β    α |     and s_0 = | 0         |   z
        | 0 0    0     1 |               | 2β        |   z_dot
        | 0 0    0     1 |               | sin(0)    |   y
        | 0 0 -(ωλ)²   0 |               | ωλ cos(0) |   y_dot
    s_t = e^tA @ s_0 generates
    y = sin(ωλt)
    Then the first row generates
    z = βy² + αy = βt² + αt
    '''

    # Increase the size of the system by adding one more variable and its derivative to model the y = mx + c line
    # The last two rows of the transition matrix and the initial state models this
    lds_size = len(s0_period)
    lds_size_trend = lds_size + 2
    A_concept_period_base_trend = np.zeros((lds_size_trend, lds_size_trend))
    A_concept_period_base_trend[: lds_size, : lds_size] = A_base

    s0_concept_period_trend = np.zeros(lds_size_trend)
    s0_concept_period_trend[: lds_size] = s0_period

    # Generate low frequency sinusoid
    lmbda = 2 * np.pi / period
    omega = frequency
    A_concept_period_base_trend[-2, -1] = 1
    A_concept_period_base_trend[-1, -2] = -(omega * lmbda) ** 2
    s0_concept_period_trend[-1] = omega * lmbda * np.cos(0)
    s0_concept_period_trend[-2] = np.sin(0)

    # Add quadratic trend, at² to the periodic time series
    A_concept_period_base_trend[0, -2] = 2 * beta  # This behavior is hard to interpret. beta = 0 seems better
    A_concept_period_base_trend[0, -1] = alpha
    s0_concept_period_trend[1] = 2 * beta

    # A_concept_period_base_trend = np.zeros((4, 4))
    # s0_concept_period_trend = np.zeros(4)
    #
    # # Generate low frequency sinusoid
    # lmbda = 2 * np.pi / period
    # omega = 0.25
    # A_concept_period_base_trend[-2, -1] = 1
    # A_concept_period_base_trend[-1, -2] = -(omega * lmbda) ** 2
    # s0_concept_period_trend[-1] = omega * lmbda * np.cos(0)
    # s0_concept_period_trend[-2] = np.sin(0)
    #
    # A_concept_period_base_trend[0, -2] = 2 * beta
    # A_concept_period_base_trend[0, -1] = alpha
    # s0_concept_period_trend[1] = 2 * beta
    #
    # print(A_concept_period_base_trend, '\n')
    # print(s0_concept_period_trend)

    return A_concept_period_base_trend, s0_concept_period_trend


def fit_periodic_model_via_fourier_decomposition(head_nodes):
    # Group seasonal head nodes according to their seasonality.
    period_to_head_nodes = {}

    for hn in head_nodes:
        if hn.period in period_to_head_nodes:
            period_to_head_nodes[hn.period].append(hn)
        else:
            period_to_head_nodes[hn.period] = [hn]

    fourier_frequency_set = set()

    for period, hns in period_to_head_nodes.items():
        # The maximum number of components we are going to evaluate
        # in search of the best number of components to be used.
        # max_k < period / 2 (Nyquist theorem)
        max_k = period // 2

        # Generate the maximum number of sinusoidal frequencies needed for the
        # period. By the Nyquist theorem this number is floor(period / 2)
        # The actual number of frequencies used to model a concept could be
        # less than this, which is decided by computing the root mean squared
        # error of the predictions for each number of components
        period_freqs = generate_frequencies_for_period(period, max_k)

        # Assemble the sinusoidal generating LDS with the maximum number
        # of components. This includes all the sinusoidals needed for
        # every lesser number of components we are going to try out.
        A_sin_max_k_base, s0_sin_max_k = assemble_sinusoidal_generating_LDS(period_freqs)

        sinusoidals = generate_sinusoidal_values_for_bins(A_sin_max_k_base, s0_sin_max_k, period)
        # for k in range(max_k):
        #     sns.lineplot(x=range(len(sinusoidals[:, 2*k])), y=sinusoidals[:, 2*k])
        #     sns.lineplot(x=range(len(sinusoidals[:, 2*k+1])), y=sinusoidals[:, 2*k+1])
        #     # sns.lineplot(x=range(len(sinusoidals[2*k, :])), y=sinusoidals[2*k, :])
        #     # sns.lineplot(x=range(len(sinusoidals[2*k+1, :])), y=sinusoidals[2*k+1, :])
        #     plt.show()
        #     plt.close()
        # continue

        # Assign transition matrix rows to head nodes with this period.
        # NOTE: At this moment we do not need to reformat the head node vector
        #       this way. We could just use the head node vector as it is and
        #       compute the transition matrix row based on the index at which
        #       each head node is at.
        #       I am doing this to make assemble_head_node_modeling_LDS()
        #       method more generalized so that I could reuse it to assemble
        #       the final complete LDS with all the nodes (seasonal head nodes
        #       with different periods and body nodes).
        #       At the moment, in the complete system, head nodes does not
        #       occupy a contiguous range of rows in the transition matrix.
        hn_to_mat_row = {}
        for v in range(0, len(hns)):
            hn_to_mat_row[hns[v]] = 2 * v

            # Again in the light of reusing the LDS
            # assembly code to assemble the complete LDS
            # hn = head_nodes[hn_ids[v]]
            hn.fourier_freqs = period_freqs

        for components in range(max_k, max_k + 1):
            for hn in hns:
                # Again in the light of reusing the LDS
                # assembly code to assemble the complete LDS
                # hn = head_nodes[hn_ids[v]]
                hn.n_components = components

            compute_fourier_coefficients_from_least_square_optimization(sinusoidals, components, hns)

            # Assemble the LDS that generates the head nodes with this
            # period using this number of components.
            A_concept_period_base, s0_concept_period = assemble_head_node_modeling_LDS(A_sin_max_k_base,
                                                                                       s0_sin_max_k,
                                                                                       components,
                                                                                       len(hn_to_mat_row),
                                                                                       hn_to_mat_row)

            # print(A_concept_period_base)
            # A_concept_period_base[0][0] = 0.07
            # A_concept_period_base[0][1] = 0.07

            # A_concept_period_base, s0_concept_period = add_linear_trend(A_concept_period_base, s0_concept_period,
            #                                                             slope=slope)
            # A_concept_period_base, s0_concept_period = add_quadratic_trend(A_concept_period_base, s0_concept_period,
            #                                                                a=0.025)
            # A_concept_period_base, s0_concept_period = add_exponential_trend(A_concept_period_base, s0_concept_period,
            #                                                                  a=0.05)
            # A_concept_period_base, s0_concept_period = add_quadratic_and_linear_trend(A_concept_period_base,
            #                                                                           s0_concept_period,
            #                                                                           alpha=1, beta=0.025)
            A_concept_period_base, s0_concept_period = add_sinusoidal_trend(A_concept_period_base, s0_concept_period,
                                                                            alpha=20, beta=0,
                                                                            period=6, frequency=0.25)
            '''
            lds_size = len(s0_concept_period)
            lds_size_trend = lds_size + 2
            A_concept_period_base_trend = np.zeros((lds_size_trend, lds_size_trend))
            A_concept_period_base_trend[: lds_size, : lds_size] = A_concept_period_base
            A_concept_period_base_trend[0][-1] = slope

            # Empirically setting these matirx positions (and with or without a slope)
            # provides exponential growth. But I cannot figure out the math and the rational.
            # A_concept_period_base_trend[0][-2] = 0.02
            # To construct y = x + c, uncomment this line. But it has no effect on the periodic time series.
            # A_concept_period_base_trend[-2][-1] = 1

            # print(A_concept_period_base_trend)

            s0_concept_period_trend = np.zeros(lds_size_trend)
            s0_concept_period_trend[: lds_size] = s0_concept_period
            # s0_concept_period_trend[-2] = 2000  # Initial value (intercept c) of y = x + c
            s0_concept_period_trend[-1] = 1
            preds = evolve_LDS(A_concept_period_base_trend, s0_concept_period_trend, period * pred_periods, pred_step)
            return preds
            '''

            preds = evolve_LDS(A_concept_period_base, s0_concept_period, period * pred_periods, pred_step)
            return preds
            # sns.lineplot(x=range(len(preds[0, :])), y=preds[0, :])
            # plt.show()
            # plt.close()




# Generating an quadratic curve
# y = 0.5x² + 2.5x + 10

"""
| 1 0 γ 0 |   | α |
| 0 1 0 0 | @ | 0 |
| 0 0 1 1 |   | 0 |
| 0 0 0 1 |   | β |
y = (½ γ²) x² + (½ γ + β) x + α
"""

'''
lds_size = 4
tot_steps = 500
A_step = np.identity(lds_size)
A_step[0, -2] = 1
A_step[0, -1] = 2
A_step[-2, -1] = 0.1
_s0 = np.zeros(lds_size)
_s0[-1] = 1

print(A_step, '\n')
print(_s0)

# A matrix to accumulate predictions of the system
preds = np.zeros((lds_size, tot_steps))
preds[:, 0] = _s0

# Evolve the LDS one step_size at a time for desired number of steps
for col in range(1, tot_steps):
    preds[:, col] = A_step @ preds[:, col - 1]
fig, ax = plt.subplots(dpi=250, figsize=(14, 7.5))
sns.lineplot(x=range(tot_steps), y=preds[0, :])
plt.tight_layout()
plt.show()
plt.close()
exit()
'''


df_rain = pd.read_csv('../../uai_2023/data/synthetic/fourier/6_Fourier_period=12_pure-std=6.142_noise-std=1.228_noise-factor=5.csv')
rain = np.array(df_rain['Noisy'].tolist())
months = np.array(range(len(rain)))

name = 'Precipitation'
n = Periodic_Time_Series(0, 12, name)
n.put_data(rain, months)

name = 'Precipitation Bin Means'
n_mean = Periodic_Time_Series(0, 12, name)
n_mean.put_data(n.bin_means, np.array(range(len(n.bin_means))))

preds = fit_periodic_model_via_fourier_decomposition(head_nodes=[n])
preds_mean = fit_periodic_model_via_fourier_decomposition(head_nodes=[n_mean])

# print(len(preds[0, :]), len(np.arange(0, 12, 0.1)))
fig, ax = plt.subplots(dpi=250, figsize=(14, 7.5))
sns.lineplot(x=months[: 12 * 3], y=rain[: 12 * 3], label='Data')
pred_x = np.arange(0, 12 * pred_periods, pred_step)
sns.lineplot(x=pred_x, y=preds[0, :], label='Predictions')
# pd.DataFrame({'t': pred_x, 'preds': preds[0, :]}).to_csv('preds.csv', index=False)
sns.lineplot(x=pred_x, y=preds[-2, :], label='Trend generation curve')  # Trend generation curve
# sns.lineplot(x=pred_x, y=preds_mean[0, :])
# sns.lineplot(x=pred_x, y=[slope * x for x in pred_x])  # Linear trend of the periodic time series
# sns.lineplot(x=pred_x, y=[0.025 * x ** 2 for x in pred_x])  # Quadratic trend of the periodic time series
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# print(list(zip(n.time_steps, n.data)))
# print()
# print(n.partitioned_data)

