from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams.update({'font.size': 18, 'figure.dpi': 150})
sns.set(rc={"lines.linewidth": 0.7})

out_dir = 'plots/timing_dist_comp/'
if out_dir:
    out_path = Path(out_dir)
    if not out_path.is_dir():
        print(f'\nMaking output directory: {out_dir}')
        out_path.mkdir(parents=True, exist_ok=True)


def plot_micro_timing_min_cag_distributions(df_mcmc_and_kde, measurement='Wall Clock Time (ns)', line=True, separate=True, summary=False):
    min_cag = df_mcmc_and_kde['Edges'] == (df_mcmc_and_kde['Nodes'] - 1)
    df_min_cag = df_mcmc_and_kde[min_cag]

    plot_no = 1
    if line:
        sns.lineplot(data=df_min_cag, x='Nodes', y=measurement, hue='Sample Type', marker='o', linewidth=2)
        if summary:
            plt.title('Percentage Speedup for a Single MCMC Iteration', size=16)
        else:
            plt.title('Micro Timing for Minimum Size CAGs', size=16)
        plt.tight_layout()
        file_name_modifier = 'line'
    else:
        if separate:
            g = sns.FacetGrid(df_min_cag, col='Sample Type', row='Nodes', hue='Sample Type', sharex='col', margin_titles=True)
            file_name_modifier = 'dist_sep'
        else:
            g = sns.FacetGrid(df_min_cag, row='Nodes', hue='Sample Type', sharex='col', margin_titles=True)
            file_name_modifier = 'dist_comb'

        g.map(sns.histplot, measurement)

        g.fig.set_figwidth(24)
        g.fig.set_figheight(11)
        g.set_titles(col_template='{col_name}', row_template='{row_name} Nodes')
        g.fig.suptitle('Micro Timing for Minimum Size CAGs', size=16)
        g.fig.subplots_adjust(top=.9)

        # Iterate thorugh each axis
        for ax in g.axes.flat:
            ax.set_ylabel('Number of Samples')

        g.add_legend()

    plt.savefig(f'{out_dir}_min_cag_{file_name_modifier}_xx_prior_hist.png')
    plot_no += 1
    # plt.show()
    plt.close()


def plot_micro_timing_distributions(df_mcmc_and_kde, measurement='Wall Clock Time (ns)', separate=True):
    df_nodes = df_mcmc_and_kde.groupby(by=['Nodes'], as_index=False)

    plot_no = 1
    for nodes, df_node in df_nodes:
        if separate:
            g = sns.FacetGrid(df_node, col='Sample Type', row='Edges', hue='Sample Type', sharex='col', sharey='row', margin_titles=True)
            file_name_modifier = 'sep'
        else:
            g = sns.FacetGrid(df_node, row='Edges', hue='Sample Type', sharex='col', sharey='row', margin_titles=True)
            file_name_modifier = 'comb'

        g.map(sns.histplot, measurement)

        g.fig.set_figwidth(24)
        g.fig.set_figheight(11)
        g.set_titles(col_template='{col_name}', row_template='{row_name} Edges')
        g.fig.suptitle(f'Micro Timing for CAGs with {nodes} Nodes', size=16)
        g.fig.subplots_adjust(top=.9)

        # Iterate thorugh each axis
        for ax in g.axes.flat:
            ax.set_ylabel('Number of Samples')

        g.add_legend()
        # plt.tight_layout()
        # plt.savefig(f'{out_dir}{plot_no}_{file_name_modifier}_{nodes}.png')
        plot_no += 1
        plt.show()
        plt.close()


def plot_micro_timing_summery_per_cag_size(df_mcmc_and_kde, measurement='Wall Clock Time (ns)', separate=True):

    def edges_to_label(row):
        if row['Edges'] == row['Nodes'] - 1:
            return '$Nodes - 1$'
        elif row['Edges'] == int((row['Nodes'] - 1) * 5 / 4):
            return '$\\frac{5(Nodes - 1)}{4}$'
        elif row['Edges'] == int((row['Nodes'] - 1) * 6 / 4):
            return '$\\frac{6(Nodes - 1)}{4}$'
        elif row['Edges'] == int((row['Nodes'] - 1) * 7 / 4):
            return '$\\frac{7(Nodes - 1)}{4}$'
        elif row['Edges'] == (row['Nodes'] - 1) * 2:
            return '$2(Nodes - 1)$'

    plot_no = 1
    if separate:
        df_nodes = df_mcmc_and_kde.groupby(by=['Nodes'], as_index=False)
        for nodes, df_node in df_nodes:
            sns.lineplot(data=df_node, x='Edges', y=measurement, hue='Sample Type', marker='o', linewidth=2)
            file_name_modifier = 'sep'
            plt.title('Variation of the Percentage Speedup for a Single MCMC Iteration\nwith the Number of Edges', size=16)
            # plt.savefig(f'{out_dir}{plot_no}_{file_name_modifier}_{nodes}.png')
            plt.show()
            plt.close()
    else:
        df_mcmc_and_kde['x label'] = df_mcmc_and_kde.apply(edges_to_label, axis=1)
        sns.lineplot(data=df_mcmc_and_kde[((df_mcmc_and_kde['Sample Type'] == '$\\theta$') & (df_mcmc_and_kde[measurement] >= 0))], x='x label', y=measurement, hue='Nodes')
        file_name_modifier = 'comb'

        # plt.tight_layout()
        plt.xlabel('Edges (as a function of Nodes)')
        plt.title('Variation of the Percentage Speedup for a Single MCMC Iteration\nwith the Number of Edges', size=16)
        # plt.savefig(f'{out_dir}{plot_no}_percentage_speedup_against_edges.png')
        plt.show()
        plot_no += 1
        plt.close()


def plot_prediction_timing_min_cag(df_prediction, measurement='Wall Clock Time (ns)', line=False, separate=True):
    min_cag = df_prediction['Edges'] == (df_prediction['Nodes'] - 1)
    df_min_cag = df_prediction[min_cag]

    if line:
        sns.lineplot(data=df_min_cag, x='Nodes', y=measurement, marker='o', linewidth=2)
        plt.title('Prediction Timing for Minimum Size CAGs', size=16)
        plt.tight_layout()
    else:
        if separate:
            g = sns.FacetGrid(df_min_cag, row='Nodes', margin_titles=True)
        else:
            g = sns.FacetGrid(df_min_cag, row='Nodes', hue='Sample Type', sharex='col', margin_titles=True)

        g.map(sns.histplot, measurement)

        g.fig.set_figwidth(24)
        g.fig.set_figheight(11)
        g.set_titles(col_template='{col_name}', row_template='{row_name} Nodes')
        g.fig.suptitle('Micro Timing for Minimum Size CAGs', size=16)
        g.fig.subplots_adjust(top=.9)

        # Iterate thorugh each axis
        for ax in g.axes.flat:
            ax.set_ylabel('Number of Samples')

        g.add_legend()

    plt.show()
    plt.close()


def plot_prediction_timing_distributions(df_prediction, measurement='Wall Clock Time (ns)', separate=True):
    df_prediction = df_prediction.groupby(by=['Nodes'], as_index=False)

    for nodes, df_node in df_prediction:
        if separate:
            g = sns.FacetGrid(df_node, col='Nodes', row='Edges', hue='Nodes', sharex='col', margin_titles=True)
        else:
            g = sns.FacetGrid(df_node, row='Edges', hue='Nodes', sharex='col', margin_titles=True)

        g.map(sns.histplot, measurement)

        g.fig.set_figwidth(24)
        g.fig.set_figheight(11)
        g.set_titles(row_template='{row_name} Edges')
        g.fig.suptitle(f'Prediction Timing Distributions for CAGs with {nodes} Nodes', size=16)
        g.fig.subplots_adjust(top=.9)

        # Iterate thorugh each axis
        for ax in g.axes.flat:
            ax.set_ylabel('Number of Samples')

        g.add_legend()
        # plt.tight_layout()
        plt.show()
        plt.close()


def analyze_micro_timing_data(df, mcmc_timing=False):
    # df_summerry = df.groupby(by=['Nodes', 'Edges', 'Sample Type'], as_index=False).agg(['mean', 'median', 'std'])
    # print(df_summerry)
    # return
    df_node_edge = df.groupby(by=['Nodes'], as_index=False)

    for ne, df_ne in df_node_edge:
        # print(ne)
        # print(df_ne.columns)
        # fig, ax = plt.subplots(dpi=250, figsize=(24, 6.75))
        g = sns.FacetGrid(df_ne, col='Sample Type', row='Edges', sharex='col', margin_titles=True)
        g.map(sns.histplot, 'Time Wall')
        plt.show()
        plt.close()
        continue

        if mcmc_timing:
            df_sample_type = df_ne.groupby(by=['Sample Type'], as_index=False)

            for st, df_st in df_sample_type:
                min_cag = df_st['Edges'] == (df_st['Nodes'] - 1)
                df_min_cag = df_st[min_cag]
                # print(st)
                # print(df_st.columns)
                # continue
                # sns.lineplot(data=df_min_cag, x='Nodes', y='MCMC Wall', marker='o', linewidth=2)
                sns.histplot(df_min_cag, x='MCMC Wall', element='step',
                             color=(0.9375, 0.5, 0.5), stat='probability')
            title = 'Sampling $\\theta$ ' if st == 1 else 'Sampling derivative '
            plt.title(title + f'{ne}')
            plt.tight_layout()
            # plt.savefig(f'{out_dir}{plot_no}_{title} - line.png')
            plt.show()
            plt.close()


def assemble_micro_timing_output_files_into_df(file_name_filter, timing_type_column, timing_type_remap):
    timing_files = timing_file_folder.glob(f'*{file_name_filter}*')

    df = pd.concat(map(pd.read_csv, timing_files), ignore_index=True)
    df.drop(['Run', 'KDE Kernels'], axis=1, inplace=True)

    df['CPU Time (ns)'] = df['CPU Time (ns)'].apply(lambda ns: ns / 1000000.0)
    df['Wall Clock Time (ns)'] = df['Wall Clock Time (ns)'].apply(lambda ns: ns / 1000000.0)

    # df[timing_type_column] = df[timing_type_column].apply(lambda timing_type: timing_type_remap.get(timing_type, timing_type))

    df.rename(columns={'Wall Clock Time (ns)': 'Wall Clock Time (ms)', 'CPU Time (ns)': 'CPU Time (ms)'}, inplace=True)
    return df


timing_file_folder_path = 'timing_2021-11-23_edge_prior_hist/'
timing_types = ['mcmc', 'kde', 'prediction']
timing_types2 = {'mcmc': {0: 'Derivative', 1: '$\\theta$'}, 'kde': {10: 'KDE', 11: 'Upd Mat'}}
timing_types2 = {'micro_mcmc': {0: 'Before Derivative', 1: 'Before $\\theta$'}, 'prior_hist': {0: 'After Derivative', 1: 'After $\\theta$'}}
dfs = []

timing_file_folder = Path(timing_file_folder_path)

for file_name_filter, timing_type_remap in timing_types2.items():
    dfs.append(assemble_micro_timing_output_files_into_df(file_name_filter, 'Sample Type', timing_type_remap))
# for timing_type in timing_types:
#
#     timing_files = timing_file_folder.glob(f'*{timing_type}*')
#
#     dfs.append(pd.concat(map(pd.read_csv, timing_files), ignore_index=True))
#
#     if timing_type == 'mcmc':
#         dfs[-1]['Sample Type'] = dfs[-1]['Sample Type'].apply(lambda st: '$\\theta$' if st == 1 else 'Derivative')
#     elif timing_type == 'kde':
#         # dfs[-1]['Sample Type'] = 'KDE'
#         dfs[-1]['Sample Type'] = dfs[-1]['Sample Type'].apply(lambda st: 'KDE' if st == 10 else 'Mat Exp' if st == 11 else 'Upd Mat')


# dfs[0].rename(columns={'Wall Clock Time (ms)': 'Wall Clock Time (ms) - before', 'CPU Time (ms)': 'CPU Time (ms) - before'}, inplace=True)
# dfs[1].rename(columns={'Wall Clock Time (ms)': 'Wall Clock Time (ms) - after', 'CPU Time (ms)': 'CPU Time (ms) - after'}, inplace=True)

df_before_summary = dfs[0].groupby(by=['Nodes', 'Edges', 'Sample Type'], as_index=False).agg(wall_before_mean=('Wall Clock Time (ms)', 'mean'),
                                                                                            wall_before_median=('Wall Clock Time (ms)', 'median'),
                                                                                            wall_before_std=('Wall Clock Time (ms)', 'std'),
                                                                                            wall_before_count=('Wall Clock Time (ms)', 'count'),
                                                                                            cpu_before_mean=('CPU Time (ms)', 'mean'),
                                                                                            cpu_before_median=('CPU Time (ms)', 'median'),
                                                                                            cpu_before_std=('CPU Time (ms)', 'std'),
                                                                                            cpu_before_count=('CPU Time (ms)', 'count')
                                                                                            ).round(2)

df_after_summary = dfs[1].groupby(by=['Nodes', 'Edges', 'Sample Type'], as_index=False).agg(wall_after_mean=('Wall Clock Time (ms)', 'mean'),
                                                                                            wall_after_median=('Wall Clock Time (ms)', 'median'),
                                                                                            wall_after_std=('Wall Clock Time (ms)', 'std'),
                                                                                            wall_after_count=('Wall Clock Time (ms)', 'count'),
                                                                                            cpu_after_mean=('CPU Time (ms)', 'mean'),
                                                                                            cpu_after_median=('CPU Time (ms)', 'median'),
                                                                                            cpu_after_std=('CPU Time (ms)', 'std'),
                                                                                            cpu_after_count=('CPU Time (ms)', 'count')
                                                                                            ).round(2)
df_before_summary = pd.merge(left=df_before_summary, right=df_after_summary, on=['Nodes', 'Edges', 'Sample Type'])

timing_type_remap = {0: 'Derivative', 1: '$\\theta$'}
df_before_summary['Sample Type'] = df_before_summary['Sample Type'].apply(lambda timing_type: timing_type_remap.get(timing_type, timing_type))

df_before_summary['Mean CPU Time Diff (ms)'] = df_before_summary.apply(lambda row: row['cpu_before_mean'] - row['cpu_after_mean'], axis=1)
df_before_summary['Median CPU Time Diff (ms)'] = df_before_summary.apply(lambda row: row['cpu_before_median'] - row['cpu_after_median'], axis=1)
df_before_summary['% Speedup (Median CPU Time)'] = df_before_summary.apply(lambda row: row['Median CPU Time Diff (ms)'] * 100 / row['cpu_before_median'], axis=1)
df_before_summary['% Speedup (Mean CPU Time)'] = df_before_summary.apply(lambda row: row['Mean CPU Time Diff (ms)'] * 100 / row['cpu_before_mean'], axis=1)

df_before_summary['Mean Wall Time Diff (ms)'] = df_before_summary.apply(lambda row: row['wall_before_mean'] - row['wall_after_mean'], axis=1)
df_before_summary['Median Wall Time Diff (ms)'] = df_before_summary.apply(lambda row: row['wall_before_median'] - row['wall_after_median'], axis=1)
df_before_summary['% Speedup (Median Wall Time)'] = df_before_summary.apply(lambda row: row['Median Wall Time Diff (ms)'] * 100 / row['wall_before_median'], axis=1)
df_before_summary['% Speedup (Mean Wall Time)'] = df_before_summary.apply(lambda row: row['Mean Wall Time Diff (ms)'] * 100 / row['wall_before_mean'], axis=1)

# plot_micro_timing_min_cag_distributions(df_before_summary, measurement='% Speedup (Median CPU Time)', line=True, separate=False, summary=True)
# plot_micro_timing_summery_per_cag_size(df_before_summary, measurement='% Speedup (Median CPU Time)', separate=False)

df_before_summary.to_csv('before_summary.csv')
df_after_summary.to_csv('after_summary.csv')


dfs[0]['Sample Type'] = dfs[0]['Sample Type'].apply(lambda timing_type: timing_types2['micro_mcmc'].get(timing_type, timing_type))
dfs[1]['Sample Type'] = dfs[1]['Sample Type'].apply(lambda timing_type: timing_types2['prior_hist'].get(timing_type, timing_type))

measurements = ['Wall Clock Time (ns)', 'CPU Time (ns)', 'CPU Time (ms)', '% Speedup (Median CPU Time)']
# plot_micro_timing_distributions(pd.concat([dfs[0], dfs[1]]), measurement=measurements[2], separate=False)
plot_micro_timing_min_cag_distributions(pd.concat([dfs[0], dfs[1]]), measurement=measurements[2], line=True, separate=False)
# plot_prediction_timing_min_cag(dfs[2])
# plot_prediction_timing_distributions(dfs[2])
