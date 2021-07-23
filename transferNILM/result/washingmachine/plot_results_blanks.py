import os
import numpy as np
import matplotlib.pyplot as plt

mains_dict = {}
target_dict = {}
prediction_dict = {}
offset = 299 # 299


mains_dict[f"run_0"] = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./explain/washingmachine_test_H1.csv_mains_2_0.npy')))
target_dict[f"run_0"] = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./explain/washingmachine_test_H1.csv_gt_2_0.npy')))
runs = np.arange(0, 549, 1)
for i in runs:
    prediction_dict[f"run_{i}"] = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./explain/washingmachine_test_H1.csv_pred_2_{i}.npy')))
plt.figure()
flag_x = False
centre_index = 8585
start_index = centre_index-offset
plot_size = 599
for i in runs: # runs == obsfucation start
    if flag_x is False:
        plt.plot(mains_dict[f"run_{i}"][start_index+offset:start_index+offset+plot_size], alpha=1, label="Aggregate")
        plt.plot(target_dict[f"run_{i}"][start_index:start_index+plot_size], alpha=1, label="Target")
        flag_x = True
    plt.plot(prediction_dict[f"run_{i}"][start_index:start_index+plot_size])
plt.title(f"Obfuscation Comparison (Test Index: {centre_index})")
plt.xlabel("Obfuscation Start (Size:50)")
plt.ylabel("Prediction (Watts)")
plt.legend(loc='upper left')
plt.show()