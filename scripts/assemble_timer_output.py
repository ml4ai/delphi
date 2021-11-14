from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 18, 'figure.dpi': 150})
sns.set(rc={"lines.linewidth": 0.7})

out_dir = 'plots/timing_dist/'
if out_dir:
    out_path = Path(out_dir)
    if not out_path.is_dir():
        print(f'\nMaking output directory: {out_dir}')
        out_path.mkdir(parents=True, exist_ok=True)


def plot_micro_timing_min_cag_distributions(df_mcmc_and_kde, measurement='Wall Clock Time (ns)', line=True, separate=True):
    min_cag = df_mcmc_and_kde['Edges'] == (df_mcmc_and_kde['Nodes'] - 1)
    df_min_cag = df_mcmc_and_kde[min_cag]

    plot_no = 1
    if line:
        sns.lineplot(data=df_min_cag, x='Nodes', y=measurement, hue='Sample Type', marker='o', linewidth=2)
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

    # plt.savefig(f'{out_dir}_min_cag_{file_name_modifier}.png')
    plot_no += 1
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

timing_file_folder_path = 'timing/'
timing_types = ['mcmc', 'kde', 'prediction']
dfs = []

timing_file_folder = Path(timing_file_folder_path)

for timing_type in timing_types:
    timing_files = timing_file_folder.glob(f'*{timing_type}*')

    dfs.append(pd.concat(map(pd.read_csv, timing_files), ignore_index=True))

    if timing_type == 'mcmc':
        dfs[-1]['Sample Type'] = dfs[-1]['Sample Type'].apply(lambda st: '$\\theta$' if st == 1 else 'Derivative')
    elif timing_type == 'kde':
        # dfs[-1]['Sample Type'] = 'KDE'
        dfs[-1]['Sample Type'] = dfs[-1]['Sample Type'].apply(lambda st: 'KDE' if st == 10 else 'Mat Exp' if st == 11 else 'Upd Mat')

measurements = ['Wall Clock Time (ns)', 'CPU Time (ns)']
plot_micro_timing_distributions(pd.concat([dfs[0], dfs[1]]), measurement=measurements[1], separate=True)
# plot_micro_timing_min_cag_distributions(pd.concat([dfs[0], dfs[1]]), measurement=measurements[1], line=False, separate=False)
# plot_prediction_timing_min_cag(dfs[2])
# plot_prediction_timing_distributions(dfs[2])
