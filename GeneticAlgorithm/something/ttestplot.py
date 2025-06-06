import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ▶️ Insert your data here
chromatic_old = [9, 10, 9, 10, 8, 10, 9, 8, 9, 9, 10, 9, 9, 10, 9, 11, 10, 8, 11, 9, 10, 10, 9, 10, 9, 9, 9, 9, 10, 9]
chromatic_new = [8, 7, 10, 8, 7, 7, 9, 8, 8, 8, 7, 8, 8, 9, 8, 8, 8, 8, 8, 8, 9, 8, 9, 7, 8, 7, 8, 7, 9, 9]

print(len(chromatic_old), len(chromatic_new))

# ▶️ Perform an independent t-test
t_stat, p_val = stats.ttest_ind(chromatic_old, chromatic_new)

print("=== T-TEST ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")
if p_val < 0.05:
    print("→ Statistically significant difference (new algorithm performs better)")
else:
    print("→ No statistically significant difference")

# ▶️ Descriptive statistics
def describe(data, label):
    print(f"\n--- {label} ---")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data)}")
    print(f"Mode: {stats.mode(data, keepdims=True).mode[0]}")
    print(f"Std dev: {np.std(data):.2f}")
    print(f"Min: {np.min(data)}, Max: {np.max(data)}")

describe(chromatic_old, "Old Algorithm")
describe(chromatic_new, "New Algorithm")

# ▶️ Histogram comparison
plt.figure(figsize=(8, 5))
bins_range = range(min(chromatic_old + chromatic_new), max(chromatic_old + chromatic_new) + 2)
plt.hist(chromatic_old, bins=bins_range, alpha=0.7, label='Old', color='red')
plt.hist(chromatic_new, bins=bins_range, alpha=0.7, label='New', color='green')
plt.title("Chromatic Number Histogram (Old vs New)")
plt.xlabel("Chromatic Number")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./ttest/chromatic_histogram.png")
plt.show()

# ▶️ Boxplot comparison
plt.figure(figsize=(6, 5))
plt.boxplot([chromatic_old, chromatic_new], labels=['Old', 'New'])
plt.title("Chromatic Number Boxplot")
plt.ylabel("Chromatic Number")
plt.grid(True)
plt.tight_layout()
plt.savefig("./ttest/chromatic_boxplot.png")
plt.show()

# ▶️ Scatter plot of results per run
plt.figure(figsize=(6, 5))
plt.scatter(range(1, 31), chromatic_old, label='Old', color='red')
plt.scatter(range(1, 31), chromatic_new, label='New', color='green')
plt.title("Chromatic Number per Run")
plt.xlabel("Run Number")
plt.ylabel("Chromatic Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./ttest/chromatic_scatter.png")
plt.show()
