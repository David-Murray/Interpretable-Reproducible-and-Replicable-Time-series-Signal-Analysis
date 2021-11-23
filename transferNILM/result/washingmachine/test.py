import os
import numpy as np

runs = 2
prediction_dict = {}

for i in np.arange(0, runs-1, 1):
    prediction_dict[f"run_{i}"] = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./explain/washingmachine_test_H1.csv_pred_2_{i}.npy')))

print(len(prediction_dict[f'run_0']))