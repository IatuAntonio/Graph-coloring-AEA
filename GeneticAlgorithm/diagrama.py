import matplotlib.pyplot as plt
import numpy as np

chromatic_old = [9, 10, 9, 10, 8, 10, 9, 8, 9, 9, 10, 9, 9, 10, 9, 11, 10, 8, 11, 9, 10, 10, 9, 10, 9, 9, 9, 9, 10, 9]
chromatic_new = [8, 7, 10, 8, 7, 7, 9, 8, 8, 8, 7, 8, 8, 9, 8, 8, 8, 8, 8, 8, 9, 8, 9, 7, 8, 7, 8, 7, 9, 9]

assert len(chromatic_old) == len(chromatic_new)

runs = range(1, len(chromatic_old) + 1)

plt.figure(figsize=(10, 6))

plt.scatter(runs, chromatic_old, label='Old Algorithm', color='red', s=60)
plt.scatter(runs, chromatic_new, label='New Algorithm', color='green', s=60)

plt.title("Chromatic Number per Run (Old vs New Algorithm)")
plt.xlabel("Run Number")
plt.ylabel("Chromatic Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./ttest/chromatic_scatter_points_only.png")
plt.show()
