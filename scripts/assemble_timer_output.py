from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse

'''
How to run
python assemble_timer_output.py -b folder_before -a folder_after -d folder_output -o file_name_prefix
'''
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams.update({'font.size': 18, 'figure.dpi': 150})
sns.set(rc={"lines.linewidth": 0.7})


# https://github.com/mwaskom/seaborn/issues/915
def fixed_boxplot(x, y, *args, label=None, **kwargs):
    sns.boxplot(x=x, y=y, *args, **kwargs, labels=[label])


def plot_micro_timing_min_cag_distributions(df_mcmc_and_kde, measurement='Wall Clock Time (ns)', hue='Sample Type',
                                            line=True, separate=True, summary=False, file_name_prefix=''):
    min_cag = df_mcmc_and_kde['Edges'] == (df_mcmc_and_kde['Nodes'] - 1)
    df_min_cag = df_mcmc_and_kde[min_cag]
    df_min_cag.to_csv('min_cag.csv', index=False)

    if line:
        sns.lineplot(data=df_min_cag, x='Nodes', y=measurement, hue=hue, marker='o', linewidth=2)
        if summary:
            plt.title('Percentage Speedup for a Single MCMC Iteration (# Edges = # Nodes - 1)', size=16)
        else:
            plt.title('Timing for a Single MCMC Iteration (# Edges = # Nodes - 1)', size=16)
        plt.tight_layout()
    else:
        if separate:
            g = sns.FacetGrid(df_min_cag, col=hue, row='Nodes', hue=hue, sharex='col', margin_titles=True)
        else:
            g = sns.FacetGrid(df_min_cag, row='Nodes', hue=hue, sharex='col', margin_titles=True)
            # g = sns.FacetGrid(df_min_cag, row='Nodes', hue='Sample Type', sharex=False, sharey=False, margin_titles=True)

        g.map(sns.histplot, measurement)
        # g.map(fixed_boxplot, 'Sample Type', measurement)

        g.fig.set_figwidth(24)
        g.fig.set_figheight(11)
        g.set_titles(col_template='{col_name}', row_template='{row_name} Nodes')
        g.fig.suptitle('Micro Timing for Minimum Size CAGs', size=16)
        g.fig.subplots_adjust(top=.9)

        # Iterate thorugh each axis
        for ax in g.axes.flat:
            ax.set_ylabel('Number of Samples')

        g.add_legend()

    if file_name_prefix:
        plt.savefig(file_name_prefix)
    else:
        plt.show()

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


def plot_micro_timing_summery_per_cag_size(df_mcmc_and_kde, measurement='Wall Clock Time (ns)', separate=True,
                                           title_specifier='', y_label='', file_name_prefix='', timing_type=''):

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

    order = ['$Nodes - 1$', '$\\frac{5(Nodes - 1)}{4}$', '$\\frac{6(Nodes - 1)}{4}$',
             '$\\frac{7(Nodes - 1)}{4}$', '$2(Nodes - 1)$']

    if separate:
        df_nodes = df_mcmc_and_kde.groupby(by=['Nodes'], as_index=False)
        for nodes, df_node in df_nodes:
            sns.lineplot(data=df_node, x='Edges', y=measurement, hue='Sample Type', marker='o', linewidth=2)
            plt.title(f'Variation of the {title_specifier} for a Single MCMC Iteration\nwith the Number of Edges for {nodes} Nodes', size=16)

            if file_name_prefix:
                plt.savefig(f'{file_name_prefix}with_num_edges_for_{nodes}_nodes.png')
            else:
                plt.show()
            plt.close()
    else:
        df_mcmc_and_kde = df_mcmc_and_kde.copy()
        df_mcmc_and_kde['x label'] = df_mcmc_and_kde.apply(edges_to_label, axis=1)
        # set categorical order
        df_mcmc_and_kde['x label'] = pd.Categorical(df_mcmc_and_kde['x label'], categories=order, ordered=True)
        df_mcmc_and_kde['Nodes'] = df_mcmc_and_kde['Nodes'].apply(lambda nodes: str(nodes))
        sns.lineplot(data=df_mcmc_and_kde[((df_mcmc_and_kde['Sample Type'] == timing_type) & (df_mcmc_and_kde[measurement] >= 0))],
                     x='x label', y=measurement, hue='Nodes', linewidth=2)

        plt.xlabel('Edges (as a function of Nodes)')
        plt.ylabel(y_label)
        plt.title(f'Variation of the {title_specifier} for a Single MCMC Iteration\nwith the Number of Edges', size=16)

        if file_name_prefix:
            plt.savefig(f'{file_name_prefix}with_num_edges.png')
        else:
            plt.show()
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


def assemble_micro_timing_output_files_into_df(folder, file_name_filter, ns_to_ms=True):
    csv_files = Path(folder).glob(f'*{file_name_filter}*.csv')

    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)
    df.drop(['Run', 'KDE Kernels'], axis=1, inplace=True, errors='ignore')

    if ns_to_ms:
        df['CPU Time (ns)'] = df['CPU Time (ns)'].apply(lambda ns: ns / 1000000.0)
        df['Wall Clock Time (ns)'] = df['Wall Clock Time (ns)'].apply(lambda ns: ns / 1000000.0)
        df.rename(columns={'Wall Clock Time (ns)': 'Wall Clock Time (ms)', 'CPU Time (ns)': 'CPU Time (ms)'},
                  inplace=True)

    return df


def combine_before_and_after_dfs(df_bf, df_af):

    def add_percentage_speedup_columns(df, timing_type, col_before, col_after):
        df[f'{timing_type} Diff (ms)'] = df.apply(lambda row: row[col_before] - row[col_after], axis=1)
        df[f'% Speedup ({timing_type})'] = df\
            .apply(lambda row: row[f'{timing_type} Diff (ms)'] * 100 / row[col_before], axis=1)

    df_summary_bf = df_bf.groupby(by=['Nodes', 'Edges', 'Sample Type'], as_index=False)\
                                                    .agg(wall_before_mean=('Wall Clock Time (ms)', 'mean'),
                                                         wall_before_median=('Wall Clock Time (ms)', 'median'),
                                                         wall_before_std=('Wall Clock Time (ms)', 'std'),
                                                         wall_before_count=('Wall Clock Time (ms)', 'count'),
                                                         cpu_before_mean=('CPU Time (ms)', 'mean'),
                                                         cpu_before_median=('CPU Time (ms)', 'median'),
                                                         cpu_before_std=('CPU Time (ms)', 'std'),
                                                         cpu_before_count=('CPU Time (ms)', 'count')
                                                         ).round(2)
    df_summary_af = df_af.groupby(by=['Nodes', 'Edges', 'Sample Type'], as_index=False)\
                                                    .agg(wall_after_mean=('Wall Clock Time (ms)', 'mean'),
                                                         wall_after_median=('Wall Clock Time (ms)', 'median'),
                                                         wall_after_std=('Wall Clock Time (ms)', 'std'),
                                                         wall_after_count=('Wall Clock Time (ms)', 'count'),
                                                         cpu_after_mean=('CPU Time (ms)', 'mean'),
                                                         cpu_after_median=('CPU Time (ms)', 'median'),
                                                         cpu_after_std=('CPU Time (ms)', 'std'),
                                                         cpu_after_count=('CPU Time (ms)', 'count')
                                                         ).round(2)

    df_both = pd.merge(left=df_summary_bf, right=df_summary_af, on=['Nodes', 'Edges', 'Sample Type'])

    df_theta = df_both[df_both['Sample Type'] == 1].copy()
    df_deri = df_both[df_both['Sample Type'] == 0].copy()

    df_theta.rename(columns={'wall_before_mean': 'theta_wall_before_mean',
                             'wall_before_median': 'theta_wall_before_median',
                             'wall_before_std': 'theta_wall_before_std',
                             'wall_before_count': 'theta_wall_before_count',
                             'cpu_before_mean': 'theta_cpu_before_mean',
                             'cpu_before_median': 'theta_cpu_before_median',
                             'cpu_before_std': 'theta_cpu_before_std',
                             'cpu_before_count': 'theta_cpu_before_count',
                             'wall_after_mean': 'theta_wall_after_mean',
                             'wall_after_median': 'theta_wall_after_median',
                             'wall_after_std': 'theta_wall_after_std',
                             'wall_after_count': 'theta_wall_after_count',
                             'cpu_after_mean': 'theta_cpu_after_mean',
                             'cpu_after_median': 'theta_cpu_after_median',
                             'cpu_after_std': 'theta_cpu_after_std',
                             'cpu_after_count': 'theta_cpu_after_count',
                             }, inplace=True)
    df_deri.rename(columns={'wall_before_mean': 'deri_wall_before_mean',
                            'wall_before_median': 'deri_wall_before_median',
                            'wall_before_std': 'deri_wall_before_std',
                            'wall_before_count': 'deri_wall_before_count',
                            'cpu_before_mean': 'deri_cpu_before_mean',
                            'cpu_before_median': 'deri_cpu_before_median',
                            'cpu_before_std': 'deri_cpu_before_std',
                            'cpu_before_count': 'deri_cpu_before_count',
                            'wall_after_mean': 'deri_wall_after_mean',
                            'wall_after_median': 'deri_wall_after_median',
                            'wall_after_std': 'deri_wall_after_std',
                            'wall_after_count': 'deri_wall_after_count',
                            'cpu_after_mean': 'deri_cpu_after_mean',
                            'cpu_after_median': 'deri_cpu_after_median',
                            'cpu_after_std': 'deri_cpu_after_std',
                            'cpu_after_count': 'deri_cpu_after_count',
                            }, inplace=True)

    df_theta.drop(['Sample Type'], axis=1, inplace=True)
    df_deri.drop(['Sample Type'], axis=1, inplace=True)

    df_average = pd.merge(left=df_theta, right=df_deri, on=['Nodes', 'Edges'])

    df_average['Average of Mean Wall Times (ms) - Before'] = df_average \
        .apply(lambda row: (row['theta_wall_before_mean'] + row['deri_wall_before_mean']) / 2, axis=1)
    df_average['Average of Median Wall Times (ms) - Before'] = df_average \
        .apply(lambda row: (row['theta_wall_before_median'] + row['deri_wall_before_median']) / 2, axis=1)
    df_average['Average of Mean CPU Times (ms) - Before'] = df_average \
        .apply(lambda row: (row['theta_cpu_before_mean'] + row['deri_cpu_before_mean']) / 2, axis=1)
    df_average['Average of Median CPU Times (ms) - Before'] = df_average \
        .apply(lambda row: (row['theta_cpu_before_median'] + row['deri_cpu_before_median']) / 2, axis=1)
    df_average['Average of Mean Wall Times (ms) - After'] = df_average \
        .apply(lambda row: (row['theta_wall_after_mean'] + row['deri_wall_after_mean']) / 2, axis=1)
    df_average['Average of Median Wall Times (ms) - After'] = df_average \
        .apply(lambda row: (row['theta_wall_after_median'] + row['deri_wall_after_median']) / 2, axis=1)
    df_average['Average of Mean CPU Times (ms) - After'] = df_average \
        .apply(lambda row: (row['theta_cpu_after_mean'] + row['deri_cpu_after_mean']) / 2, axis=1)
    df_average['Average of Median CPU Times (ms) - After'] = df_average \
        .apply(lambda row: (row['theta_cpu_after_median'] + row['deri_cpu_after_median']) / 2, axis=1)

    add_percentage_speedup_columns(df_average, 'Average of Mean CPU Times',
                                   'Average of Mean Wall Times (ms) - Before',
                                   'Average of Mean Wall Times (ms) - After')
    add_percentage_speedup_columns(df_average, 'Average of Median CPU Times',
                                   'Average of Median Wall Times (ms) - Before',
                                   'Average of Median Wall Times (ms) - After')
    add_percentage_speedup_columns(df_average, 'Average of Mean Wall Times',
                                   'Average of Mean CPU Times (ms) - Before',
                                   'Average of Mean CPU Times (ms) - After')
    add_percentage_speedup_columns(df_average, 'Average of Median Wall Times',
                                   'Average of Median CPU Times (ms) - Before',
                                   'Average of Median CPU Times (ms) - After')

    value_vars = ['Average of Mean Wall Times (ms) - Before',   'Average of Mean Wall Times (ms) - After',
                  'Average of Median Wall Times (ms) - Before', 'Average of Median Wall Times (ms) - After',
                  'Average of Mean CPU Times (ms) - Before',    'Average of Mean CPU Times (ms) - After',
                  'Average of Median CPU Times (ms) - Before',  'Average of Median CPU Times (ms) - After',
                  'Average of Mean CPU Times Diff (ms)',     '% Speedup (Average of Mean CPU Times)',
                  'Average of Median CPU Times Diff (ms)',   '% Speedup (Average of Median CPU Times)',
                  'Average of Mean Wall Times Diff (ms)',    '% Speedup (Average of Mean Wall Times)',
                  'Average of Median Wall Times Diff (ms)',  '% Speedup (Average of Median Wall Times)']
    df_average = pd.melt(df_average, id_vars=['Nodes', 'Edges'],
                         value_vars=value_vars, value_name='Average Timing', var_name='Timing Type')

    add_percentage_speedup_columns(df_both, 'Mean CPU Time', 'cpu_before_mean', 'cpu_after_mean')
    add_percentage_speedup_columns(df_both, 'Median CPU Time', 'cpu_before_median', 'cpu_after_median')
    add_percentage_speedup_columns(df_both, 'Mean Wall Time', 'wall_before_mean', 'wall_after_mean')
    add_percentage_speedup_columns(df_both, 'Median Wall Time', 'wall_before_median', 'wall_after_median')

    return df_both, df_average


def plot_micro_timing_min_cag_averages(df_average, timing_type, speedup=True, save=False, file_name_prefix=''):
    min_cag = df_average['Edges'] == (df_average['Nodes'] - 1)
    df_min_cag = df_average[min_cag]

    if speedup:
        filter = df_min_cag['Timing Type'] == f'% Speedup ({timing_type})'
        title = 'Average Percentage Speedup for a Single MCMC Iteration (# Edges = # Nodes - 1)'
        y_label = 'Average Percentage Speedup'
        file_name = f'{file_name_prefix}avg_percent_speedup.png'
    else:
        filter = (df_min_cag['Timing Type'] == f'{timing_type} (ms) - Before') | \
                 (df_min_cag['Timing Type'] == f'{timing_type} (ms) - After')
        title = 'Average Timing for a Single MCMC Iteration (# Edges = # Nodes - 1)'
        y_label = 'Average Timing (ms)'
        file_name = f'{file_name_prefix}avg_timing_before_after.png'

    df_min_cag = df_min_cag[filter]

    sns.lineplot(data=df_min_cag, x='Nodes', y='Average Timing', hue='Timing Type', marker='o', linewidth=2)
    plt.title(title)
    plt.ylabel(y_label)
    plt.tight_layout()
    if save:
        plt.savefig(file_name)
    else:
        plt.show()
    plt.close()


def rename_sample_type(df):
    timing_type_remap = {0: 'Derivative',
                         1: '$\\theta$',
                         10: 'KDE',
                         11: 'Mat Exp',
                         12: 'Update TM',
                         13: 'LL Calc'}
    df['Sample Type'] = df['Sample Type'] \
        .apply(lambda timing_type: timing_type_remap.get(timing_type, timing_type))


parser = argparse.ArgumentParser(description='Plot Delphi speedup timing results before and after an optimization')
parser.add_argument('-b', metavar='Before timing results directory', type=str, default='before',
                    help='Directory where timing results before the optimization are kept')
parser.add_argument('-a', metavar='After timing results directory', type=str, default='after',
                    help='Directory where timing results after the optimization are kept')
parser.add_argument('-fb', metavar='Before file name filter', type=str, default='mcmc',
                    help='File name specifier to filter files in the before directory')
parser.add_argument('-fa', metavar='After file name filter', type=str, default='mcmc',
                    help='File name specifier to filter files in the after directory')
parser.add_argument('-d', metavar='Output directory name', type=str, default='timing_plots',
                    help='Directory where output plots are saved')
parser.add_argument('-o', metavar='Output file name specifier', type=str, default='timing',
                    help='This specifier will be prefixed to all the output files')

args = parser.parse_args()

timing_folder_before = args.b + '/'
timing_folder_after = args.a + '/'
file_name_filter_before = args.fb
file_name_filter_after = args.fa
out_dir = args.d + '/'
out_file_name_prefix = args.o

if out_dir:
    out_path = Path(out_dir)
    if not out_path.is_dir():
        print(f'\nMaking output directory: {out_dir}')
        out_path.mkdir(parents=True, exist_ok=True)

df_before = assemble_micro_timing_output_files_into_df(timing_folder_before, file_name_filter_before)
df_after = assemble_micro_timing_output_files_into_df(timing_folder_after, file_name_filter_after)

df_after.to_csv(f'{out_dir}{out_file_name_prefix}_after.csv', index=False)
df_before.to_csv(f'{out_dir}{out_file_name_prefix}_before.csv', index=False)

df_all, df_avg = combine_before_and_after_dfs(df_before, df_after)

rename_sample_type(df_all)

df_all.to_csv(f'{out_dir}{out_file_name_prefix}_timing_summary.csv', index=False)
df_avg.to_csv(f'{out_dir}{out_file_name_prefix}_timing_average.csv', index=False)

plot_micro_timing_min_cag_distributions(df_all, measurement='% Speedup (Median CPU Time)', line=True, separate=False,
                                        summary=True,
                                        file_name_prefix=f'{out_dir}{out_file_name_prefix}_theta_vs_derivative')

plot_micro_timing_min_cag_averages(df_avg, 'Average of Median CPU Times', speedup=False, save=True,
                                   file_name_prefix=f'{out_dir}{out_file_name_prefix}_median_')
plot_micro_timing_min_cag_averages(df_avg, 'Average of Median CPU Times', speedup=True, save=True,
                                   file_name_prefix=f'{out_dir}{out_file_name_prefix}_median_')
plot_micro_timing_min_cag_averages(df_avg, 'Average of Mean CPU Times', speedup=False, save=True,
                                   file_name_prefix=f'{out_dir}{out_file_name_prefix}_mean_')
plot_micro_timing_min_cag_averages(df_avg, 'Average of Mean CPU Times', speedup=True, save=True,
                                   file_name_prefix=f'{out_dir}{out_file_name_prefix}_mean_')

df_after_embedded = assemble_micro_timing_output_files_into_df(timing_folder_after, file_name_filter='embeded')
df_kde = assemble_micro_timing_output_files_into_df(timing_folder_after, file_name_filter='kde')

rename_sample_type(df_after_embedded)
rename_sample_type(df_kde)
rename_sample_type(df_after)

plot_micro_timing_summery_per_cag_size(df_all, measurement='cpu_after_median', separate=False,
                                       title_specifier='Median Duration', y_label='Median Duration (ms)',
                                       file_name_prefix=f'{out_dir}{out_file_name_prefix}_', timing_type='$\\theta$')

for component in ['KDE', 'Mat Exp', 'Update TM', 'LL Calc']:
    plot_micro_timing_summery_per_cag_size(df_after_embedded, measurement='CPU Time (ms)', separate=False,
                                           title_specifier=f'{component} Duration', y_label='CPU Time (ms)',
                                           file_name_prefix=f'{out_dir}{out_file_name_prefix}_{component}_', timing_type=component)
plot_micro_timing_summery_per_cag_size(df_after_embedded, measurement='CPU Time (ms)', separate=True,
                                       title_specifier='Median Duration', y_label='Median Duration (ms)',
                                       file_name_prefix=f'{out_dir}{out_file_name_prefix}_')

df_after_embedded.rename(columns={'Sample Type': 'Timing Type'}, inplace=True)
df_kde.rename(columns={'Sample Type': 'Timing Type'}, inplace=True)
df_after.rename(columns={'Sample Type': 'Timing Type'}, inplace=True)

timing_type_remap = {'$\\theta$': '$\\theta$ = KDE + Mat Exp + Update TM + LL Calc'}
df_after_embedded['Timing Type'] = df_after_embedded['Timing Type'] \
    .apply(lambda timing_type: timing_type_remap.get(timing_type, timing_type))
df_after['Timing Type'] = df_after['Timing Type'] \
    .apply(lambda timing_type: timing_type_remap.get(timing_type, timing_type))

plot_micro_timing_min_cag_distributions(pd.concat([df_kde, df_after]), measurement='CPU Time (ms)', line=True,
                                        separate=False, summary=False, hue='Timing Type',
                                        file_name_prefix=f'{out_dir}{out_file_name_prefix}_mcmc_components_profiler')
plot_micro_timing_min_cag_distributions(pd.concat([df_after_embedded, df_after]), measurement='CPU Time (ms)', line=True,
                                        separate=False, summary=False, hue='Timing Type',
                                        file_name_prefix=f'{out_dir}{out_file_name_prefix}_mcmc_components_embedded')

# plot_micro_timing_min_cag_distributions(df_after_embedded[df_after_embedded['Sample Type'] == 'Update TM'], measurement='CPU Time (ms)',
#                                         line=True, separate=False, summary=False, hue='Timing Type')
#                                         file_name=f'{out_dir}min_cag_percent_speedups.png')

# plot_micro_timing_summery_per_cag_size(df_all, measurement='% Speedup (Median CPU Time)', separate=False)
