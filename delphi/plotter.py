import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import seaborn as sns
import math

sns.set()


# https://gist.github.com/Raudcu/44b43c7f3f893fe2f4dd900afb740b7f
class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title

# Plots the complete state of a delphi model.
# There is a lot of repeated code here.
def delphi_plotter(model_state, num_bins=400, rotation=45,
        out_dir='plots', file_name_prefix=''):

    if out_dir:
        out_path = pathlib.Path(out_dir)
        if not out_path.is_dir():
            print(f'\nMaking output directory: {out_dir}')
            out_path.mkdir(parents=True, exist_ok=True)

    concept_indicators, edges, adjectives, polarities, edge_data, derivatives, data_range, data_set, pred_range, predictions, cis  = model_state

    plot_num = 1

    # Plot theta prior and sample distributions
    for idx, thetas in enumerate(edge_data):
        df_prior = pd.DataFrame({'Prior': thetas[0]})
        df_prior = pd.melt(df_prior, value_vars =['Prior'], value_name='Theta', var_name='Prior_or_Sample')

        df_theta_samples = pd.DataFrame({'Sampled Thetas': thetas[1]})
        df_theta_samples = pd.melt(df_theta_samples, value_vars =['Sampled Thetas'], value_name='Theta', var_name='Prior_or_Sample')
        df_theta_samples_grp = df_theta_samples.groupby(by=['Theta'], as_index=False).count()
        df_theta_samples_grp.rename(columns={'Prior_or_Sample': '# of Samples'},
                inplace=True)

        bin_bounds = np.histogram_bin_edges(thetas[0] + thetas[1], bins=num_bins)

        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=150, figsize=(8, 4.5))
        sns.set_style("white")
        ax3 = ax2.twinx()

        sns.histplot(df_theta_samples, x='Theta', element='step',
                color=(0.9375, 0.5, 0.5), ax=ax3, stat='probability',
                bins=bin_bounds)
        sns.histplot(df_prior, x='Theta', element='step',
                color=(0.25, 0.875, 0.8125, 0.5), ax=ax2, stat='probability',
                bins=bin_bounds)

        df_theta_samples_grp['Theta'] = df_theta_samples_grp['Theta'].apply(lambda x: round(x, 3))
        g = sns.barplot(x=df_theta_samples_grp['Theta'],
                y=df_theta_samples_grp['# of Samples'],
                color=(0.9375, 0.5, 0.5), ax=ax1)

        source = edges[idx][0].split('/')[-1]
        target = edges[idx][1].split('/')[-1]

        adj = adjectives[idx]
        pol = polarities[idx]

        fig.suptitle(f'({source})' + r'$\longrightarrow$' + f'({target})\n({adj[0]}, {pol[0]})' +
                r'$\longrightarrow$' + f'({adj[1]}, {pol[1]})')
        ax1.set_title('Theta Samples')
        ax2.set_title('Theta Prior and Samples')
        ax2.set_ylabel('Prior\nProbabilities')
        ax3.set_ylabel('Samples\nProbabilities')
        g.set_xticklabels(g.get_xticklabels(), rotation=rotation)
        plt.tight_layout()

        if out_dir:
            plt.savefig(f'{out_dir}/{file_name_prefix}_{plot_num}_Thetas_{source}--{target}.png')
            plot_num += 1
        else:
            plt.show()
        plt.close()


    # Plot sampled derivatives
    df_derivatives = pd.DataFrame.from_dict(derivatives)
    df_derivatives = pd.melt(df_derivatives, value_vars=df_derivatives.columns,
            value_name='Derivative', var_name='Concept')
    df_derivatives['Concept'] = df_derivatives['Concept'].apply(lambda x:
            x.split('/')[-1])
    df_derivatives['Derivative'] = df_derivatives['Derivative'].apply(lambda x:
            round(x, 3))
    df_derivatives['# of Samples'] = df_derivatives['Concept'].apply(lambda x:
            1)
    df_derivatives_grp = df_derivatives.groupby(by=['Concept', 'Derivative'], as_index=False).count()

    g = sns.FacetGrid(df_derivatives_grp, col='Concept', sharex=False,
            sharey=False,
            col_wrap=math.ceil(math.sqrt(len(df_derivatives.columns))))
    g.map(sns.barplot, 'Derivative', '# of Samples')
    '''
    g = sns.FacetGrid(df_derivatives, col='Concept', sharex=False,
            sharey=False,
            col_wrap=math.ceil(math.sqrt(len(df_derivatives.columns))))
    g.map(sns.histplot, 'Derivative', element='step', kde=True)
    '''

    g.set_axis_labels('Derivative', '# of Samples')
    for ax in g.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    g.fig.suptitle('Sampled Initial Derivatives')
    g.tight_layout()

    if out_dir:
        plt.savefig(f'{out_dir}/{file_name_prefix}_{plot_num}_Derivatives.png', dpi=150)
        plot_num += 1
    else:
        plt.show()
    plt.close()


    # Plot predictions from the full set of predictions (plots the mean and the
    # confidence interval) and the full data set
    for ind, preds in predictions.items():
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(dpi=150, figsize=(8, 4.5))

        df_preds = pd.DataFrame.from_dict(preds)
        df_preds= pd.melt(df_preds, value_vars=df_preds.columns,
                value_name='Prediction', var_name='Time Step')
        df_preds['Time Step'] = df_preds['Time Step'].apply(lambda ts:
                pd.to_datetime(pred_range[ts]))

        df_data = pd.DataFrame.from_dict(data_set[ind])
        df_data['Time Step'] = df_data['Time Step'].apply(lambda ts:
                pd.to_datetime(data_range[int(ts)]))

        # Aggregate multiple coinciding data points
        df_data['frequency'] = df_data['Time Step'].apply(lambda x: 1)
        df_data_grp = df_data.groupby(by=['Time Step', 'Data'], as_index=False).count()
        df_data_grp['Time Step'] = pd.to_datetime(df_data_grp['Time Step'])

        g = sns.lineplot(ax=ax, data=df_preds, x='Time Step', y='Prediction',
                sort=False, marker='D', label='Mean Prediction')

        if df_data_grp['frequency'].max() == df_data_grp['frequency'].min():
            # There are no coinciding data points
            sns.scatterplot(
                ax=ax,
                data=df_data_grp,
                y='Data',
                x='Time Step',
                marker='o',
                label='Data',
                color='red'
            )
        else:
            sns.scatterplot(
                ax=ax,
                data=df_data_grp,
                y='Data',
                x='Time Step',
                marker='o',
                label='Data',
                hue=df_data_grp['frequency'].tolist(),
                palette='ch:r=-.8, l=.75',
                size=df_data_grp['frequency'].tolist(),
                sizes=(50, 250)
            )

            handles, labels = ax.get_legend_handles_labels()
            handles.insert(2, 'Number of Data Points')
            labels.insert(2, '')
            ax.legend(handles, labels, handler_map={str: LegendTitle({'fontsize':
                12})}, fancybox=True)

        # Set x-axis tick marks
        '''
        dates = set(data_range).union(set(pred_range))
        dates = sorted(list(dates))
        ax.set_xticks(pd.to_datetime(dates))
        ax.set_xticklabels(
            pd.to_datetime(dates), rotation=45, ha="right", fontsize=8
        )
        xfmt = mdates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(xfmt)
        '''

        ind = ind.split('/')[-1]
        plt.title(f'Mean Predictions and Data\n{ind}')
        plt.tight_layout()

        if out_dir:
            plt.savefig(f'{out_dir}/{file_name_prefix}_{plot_num}_Data_and_Predictions_{ind}.png')
            plot_num += 1
        else:
            plt.show()
        plt.close()


    # Plot predictions from the summarized (median, upper and lower confidence
    # intervals) set of predictions (plots the median and the confidence
    # interval) and the full data set
    for ind, ind_cis in cis.items():
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(dpi=150, figsize=(8, 4.5))

        df_cis = pd.DataFrame.from_dict(ind_cis)
        df_cis['Time Step'] = pd.to_datetime(pred_range)

        sns.lineplot(ax=ax, data=df_cis, x='Time Step', y='Median',
                sort=False, marker='D', label='Median Prediction')
        ax.fill_between(
            x=df_cis['Time Step'],
            y1=df_cis['Upper 95% CI'],
            y2=df_cis['Lower 95% CI'],
            alpha=0.2,
            color='red',
            label='95% CI'
        )

        df_data = pd.DataFrame.from_dict(data_set[ind])
        df_data['Time Step'] = df_data['Time Step'].apply(lambda ts:
                pd.to_datetime(data_range[int(ts)]))

        # Aggregate multiple coinciding data points
        df_data['frequency'] = df_data['Time Step'].apply(lambda x: 1)
        df_data_grp = df_data.groupby(by=['Time Step', 'Data'], as_index=False).count()
        df_data_grp['Time Step'] = pd.to_datetime(df_data_grp['Time Step'])

        if df_data_grp['frequency'].max() == df_data_grp['frequency'].min():
            # There are no coinciding data points
            sns.scatterplot(
                ax=ax,
                data=df_data_grp,
                y='Data',
                x='Time Step',
                marker='o',
                label='Data',
                color='red'
            )
        else:
            sns.scatterplot(
                ax=ax,
                data=df_data_grp,
                y='Data',
                x='Time Step',
                marker='o',
                label='Data',
                hue=df_data_grp['frequency'].tolist(),
                palette='ch:r=-.8, l=.75',
                size=df_data_grp['frequency'].tolist(),
                sizes=(50, 250)
            )

            handles, labels = ax.get_legend_handles_labels()
            handles.insert(2, 'Number of Data Points')
            labels.insert(2, '')
            ax.legend(handles, labels, handler_map={str: LegendTitle({'fontsize':
                12})}, fancybox=True)


        ind = ind.split('/')[-1]
        plt.title(f'Median Predictions and $95\%$ Confidence Interval\n{ind}')

        if out_dir:
            plt.savefig(f'{out_dir}/{file_name_prefix}_{plot_num}_Predictions_Median_and_CI_{ind}.png')
            plot_num += 1
        else:
            plt.show()
        plt.close()


    # Plot predictions for pairs of indicators related by edges
    # from the summarized (median, upper and lower confidence
    # intervals) set of predictions (plots the median) and the summarized data set
    # of data (without plotting multiple coinciding data points)
    for idx, edge in enumerate(edges):
        source = edge[0]
        target = edge[1]

        adj = adjectives[idx]
        pol = polarities[idx]

        for ind_source in concept_indicators[source]:
            for ind_target in concept_indicators[target]:
                sns.set_style("whitegrid")
                fig, ax_left = plt.subplots(dpi=150, figsize=(8, 4.5))
                sns.set_style("white")
                ax_right = ax_left.twinx()

                df_source = pd.DataFrame.from_dict(cis[ind_source])
                df_target = pd.DataFrame.from_dict(cis[ind_target])

                df_source['Time Step'] = df_source.index
                df_target['Time Step'] = df_target.index

                df_source['Time Step'] = df_source['Time Step'].apply(lambda ts:
                        pd.to_datetime(pred_range[ts]))
                df_target['Time Step'] = df_target['Time Step'].apply(lambda ts:
                        pd.to_datetime(pred_range[ts]))

                df_source_data = pd.DataFrame.from_dict(data_set[ind_source])
                df_target_data = pd.DataFrame.from_dict(data_set[ind_target])

                df_source_data['Time Step'] = df_source_data['Time Step'].apply(lambda ts:
                        pd.to_datetime(data_range[int(ts)]))
                df_target_data['Time Step'] = df_target_data['Time Step'].apply(lambda ts:
                        pd.to_datetime(data_range[int(ts)]))

                color_left = 'tab:red'
                color_right = 'tab:blue'

                source = ind_source.split('/')[-1]
                target = ind_target.split('/')[-1]

                sns.lineplot(ax=ax_left, data=df_source, x='Time Step', y='Median',
                        sort=False, marker='o', color=color_left,
                        label=f'{source}')
                sns.lineplot(ax=ax_right, data=df_target, x='Time Step', y='Median',
                        sort=False, marker='o', color=color_right,
                        label=f'{target}')

                sns.scatterplot(ax=ax_left, data=df_source_data, x='Time Step',
                    y='Data', marker='x', color=color_left, label=f'{source}')
                sns.scatterplot(ax=ax_right, data=df_target_data, x='Time Step',
                        y='Data', marker='x', color=color_right, label=f'{target}')

                # Legend
                handles_left, labels_left = ax_left.get_legend_handles_labels()
                handles_right, labels_right = ax_right.get_legend_handles_labels()
                ax_right.get_legend().remove()
                handles_left.insert(2, handles_right[1])
                labels_left.insert(2, '  ')
                labels_left[1] = '  '
                handles_left.insert(1, 'Data')
                labels_left.insert(1, '')
                handles_left.insert(1, handles_right[0])
                labels_left.insert(1, labels_right[0])
                handles_left.insert(0, 'Predictions')
                labels_left.insert(0, '')
                ax_left.legend(handles_left, labels_left, fancybox=True,
                        handler_map={str: LegendTitle({'fontsize': 12})},
                        ncol=2, loc='upper center', bbox_to_anchor=(0.5,
                            -0.25), markerfirst=False)

                ax_left.set_xlabel('Time Step')
                ax_left.set_ylabel(ind_source, color=color_left)
                ax_left.tick_params(axis='y', labelcolor=color_left)

                ax_right.set_ylabel(ind_target, color=color_right)
                ax_right.tick_params(axis='y', labelcolor=color_right)

                fig.suptitle(f'({source})' +
                        r'$\longrightarrow$' + f'({target})\n({adj[0]}, {pol[0]})' +
                        r'$\longrightarrow$' + f'({adj[1]}, {pol[1]})')

                fig.tight_layout()

                if out_dir:
                    plt.savefig(f'{out_dir}/{file_name_prefix}_{plot_num}_{source}--{target}.png')
                    plot_num += 1
                else:
                    plt.show()
                plt.close()

