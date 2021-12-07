"""
Compares two snapshots of how Delphi worked at two points of its implementation
A snapshot taken before and a snapshot taken after a code change such as an
optimization or refactoring could be compared to verify whether the accuracy of the
code got changed as a byproduct of the changes made to the code. If there is no
change at all, the before and after snaps should be identical and the plots produced
by this script should be a single bar at zero.

If there are only minor changes, the plots should be concentrated at or close to zero.
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

if len(sys.argv) < 3:
    print(f'\nUsage: {sys.argv[0].split("/")[-1]} <snap before> <snap after>\n')
    exit()

df_before = pd.read_csv(sys.argv[1])
df_before.rename(columns={'MAP_ll': 'MAP_ll_before'}, inplace=True)
df_after = pd.read_csv(sys.argv[2])
df_after.rename(columns={'MAP_ll': 'MAP_ll_after'}, inplace=True)

x = 'MAP LL Difference'
df_diff = pd.merge(left=df_before, right=df_after, on=['Model', 'Seed'])
df_diff[x] = df_diff['MAP_ll_before'] - df_diff['MAP_ll_after']

# Plot log likelihoods
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=150, figsize=(8, 4.5))
plt.rcParams['font.size'] = 12

sns.histplot(df_diff, x=x, element='step',
             color=(0.25, 0.875, 0.8125, 0.5), ax=ax1, stat='probability')

df_diff_grp = df_diff.groupby(by=[x], as_index=False).count()
df_diff_grp.rename(columns={'Seed': 'Frequency'}, inplace=True)
sns.barplot(x=df_diff_grp[x], y=df_diff_grp['Frequency'],
            color=(0.9375, 0.5, 0.5), ax=ax2)

plt.suptitle(f'MAP Estimate Log Likelihood Difference\nBefore and After a Code Change:\
 $(\mu = {df_diff_grp[x].mean():.2f}, \sigma = {df_diff_grp[x].std()})$')
ax1.set_title('Probability Distribution')
ax2.set_title('Bar Plot')
ax1.set_xlabel('MAP Estimate Log Likelihood Difference')
ax2.set_xlabel('MAP Estimate Log Likelihood Difference')
ax2.set_xticklabels([f'{float(t.get_text()):.2f}' for t in ax2.get_xticklabels()], rotation=45)
plt.tight_layout()
plt.show()
