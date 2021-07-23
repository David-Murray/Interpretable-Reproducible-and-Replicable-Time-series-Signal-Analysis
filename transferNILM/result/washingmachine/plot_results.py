import os
import numpy as np
import matplotlib.pyplot as plt

mains_dict = {}
target_dict = {}
prediction_dict = {}
offset = 299
runs = ["1", "2", "3", "4", "5", "rng_1", "rng_6", "rng_3", "rng_4", "rng_5"]
mains_dict[f"run_1"] = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./reproduce/washingmachine_test_H8.csv_mains_1.npy')))
target_dict[f"run_1"] = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./reproduce/washingmachine_test_H8.csv_gt_1.npy')))
for i in runs:
    prediction_dict[f"run_{i}"] = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./reproduce/washingmachine_test_H8.csv_pred_{i}.npy')))
plt.figure(figsize=(10,6))
flag_x = False
start_index = 2600
plot_size = 800
plt.plot(mains_dict[f"run_1"][start_index+offset:start_index+offset+plot_size], alpha=1, label="Aggregate")
plt.plot(target_dict[f"run_1"][start_index:start_index+plot_size], alpha=1, label="Target")
for i in runs:
    plt.plot(prediction_dict[f"run_{i}"][start_index:start_index+plot_size], alpha=0.35, label=f"run_{i}" if i != "rng_6" else f"run_rng_2")
plt.title("Model Comparison - REFIT")
plt.xlabel("Sample", fontsize=13)
plt.ylabel("Consumption (Watts)", fontsize=13)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()