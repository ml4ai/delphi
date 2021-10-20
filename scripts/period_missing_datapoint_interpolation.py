from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 18, 'figure.dpi': 150})

## Period interpolation check
filled_months = [2, 3, 5, 9]
generated_monthly_latent_sequence_for_a_year = [0, 0, 10, 20, 0, 5, 0, 0, 0, 30, 0, 0]
print(generated_monthly_latent_sequence_for_a_year)
for i in range(len(filled_months)):
    month_start = filled_months[i]
    month_end = filled_months[(i+1) % len(filled_months)]

    num_missing_months = 0
    if month_end > month_start:
        num_missing_months = month_end - month_start - 1
    else:
        num_missing_months = (11 - month_start) + month_end

    print(month_start, '|', month_end, '|', num_missing_months)

    for month_missing in range(1, num_missing_months + 1):
        print((month_start + month_missing) % 12)
        generated_monthly_latent_sequence_for_a_year[(month_start + month_missing) % 12] = \
            ((num_missing_months - month_missing + 1) * generated_monthly_latent_sequence_for_a_year[month_start]
             + month_missing * generated_monthly_latent_sequence_for_a_year[month_end]) / (num_missing_months + 1)

print(generated_monthly_latent_sequence_for_a_year)
plt.plot(generated_monthly_latent_sequence_for_a_year, marker='o', linewidth='2')
plt.plot([0, 0, 10, 20, 0, 5, 0, 0, 0, 30, 0, 0], marker='*', linewidth='2')
plt.show()
