import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.lib.polynomial import _polyfit_dispatcher
from numpy.lib.twodim_base import diag


mains_dict = {}
target_dict = {}
prediction_dict = {}
offset = 299
runs = 550
prediction_none = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./explain/washingmachine_test_H1.csv_pred_2_None.npy')))
mains_dict[f"run_0"] = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./explain/washingmachine_test_H1.csv_mains_2_0.npy')))
target_dict[f"run_0"] = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./explain/washingmachine_test_H1.csv_gt_2_0.npy')))
for i in np.arange(0, runs-1, 1):
    prediction_dict[f"run_{i}"] = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), f'./explain/washingmachine_test_H1.csv_pred_2_{i}.npy')))
fig_1, axs_1 = plt.subplots(2, figsize=(10,6))
fig_1.suptitle('Learned Features via Obfuscation')


flag_x = False
centre_index = 575600 # 8585 # 8500 # 959750 # 152350
plot_size = 599
alpha_val = 1
blank_size = 50
obs_row = None
diag_rows = None
start_index = centre_index - offset
end_index = start_index + plot_size
axs_1[0].plot(np.arange(start_index-offset,end_index+offset), mains_dict[f"run_0"][start_index:start_index+offset+plot_size+offset], alpha=1, label="Aggregate")
axs_1[0].plot(np.arange(start_index,end_index),target_dict[f"run_0"][start_index:end_index], alpha=1, label="Target")
axs_1[0].plot(np.arange(start_index,end_index),prediction_none[start_index:end_index], alpha=1, label="Prediction (No Obs)")

target = target_dict[f"run_0"][start_index:end_index].flatten()
prediction = prediction_none[start_index:end_index].flatten()

nde = np.sum((target - prediction) ** 2) / np.sum((target ** 2))
r = np.sum(target * 8 * 1.0 / 3600.0)
rhat = np.sum(prediction * 8 * 1.0 / 3600.0)
sae = np.abs(r - rhat) / np.abs(r)
err = np.abs(target - prediction)
mae= np.mean(err)
print(f"MAE:{mae} SAE:{sae} NDE:{nde}")

for i in np.arange(0, runs-1, 1): # runs == obsfucation start
    new_stack = np.array(prediction_dict[f"run_{i}"][start_index:end_index-blank_size])
    if obs_row is None:
        obs_row = new_stack
    else:
        obs_row = np.vstack((obs_row, new_stack))
obs_row_blanks = np.fliplr(np.triu(np.fliplr(obs_row), k=-330))
obs_row_blanks = np.fliplr(np.tril(np.fliplr(obs_row_blanks), k=300))

axs_1[0].set_title(f"Obfuscation (Test Index: {centre_index})", fontsize=13)
# axs_1[0].set_xlabel("Sample", fontsize=13)
axs_1[0].set_ylabel("Prediction (Watts)", fontsize=13)
axs_1[0].legend(loc='upper right')
axs_1[0].set_xlim(start_index, end_index-20) # minus 20 samples lines up line plot with heatmap

imshow = axs_1[1].imshow(obs_row,  aspect='auto', cmap='nipy_spectral', extent=[start_index,end_index-blank_size,550,0])
imshow_blanks = axs_1[1].imshow(obs_row_blanks,  aspect='auto', cmap='nipy_spectral', extent=[start_index,end_index-blank_size,550,0])
divider = make_axes_locatable(axs_1[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig_1.colorbar(imshow, cax=cax)
cbar.set_label("Power (Watts)", fontsize=13)

# Feature Area 1
axs_1[0].axvspan(575502, 575502+50, facecolor='0.5', alpha=0.25)
axs_1[0].annotate('1.', xy=(575502+5, 2000), xycoords='data')
axs_1[1].add_patch(Ellipse((575474, 346),
        width=20,
        height=100,
        angle=0,
        linewidth=1, fill=False))
axs_1[1].annotate('1.', xy=(575466, 334), xycoords='data', fontsize=12, color="white")
# Feature Area 2
axs_1[0].axvspan(575666, 575666+50, facecolor='0.5', alpha=0.25)
axs_1[0].annotate('2.', xy=(575666+5, 2000), xycoords='data')
axs_1[1].add_patch(Ellipse((575480, 496),
        width=45,
        height=32,
        angle=0, edgecolor="white",
        linewidth=1, fill=False))
axs_1[1].annotate('2.', xy=(575480, 496), xycoords='data', fontsize=12, color="white")
# Feature Area 3
axs_1[0].axvspan(575559, 575559+50, facecolor='0.5', alpha=0.25)
axs_1[0].annotate('3.', xy=(575559+5, 2000), xycoords='data')
axs_1[1].add_patch(Ellipse((575552, 321),
        width=16,
        height=40,
        angle=0, edgecolor="white",
        linewidth=1, fill=False))
axs_1[1].annotate('3.', xy=(575532, 321), xycoords='data', fontsize=12, color="white")


axs_1[1].set_xlabel("Sample", fontsize=13)
axs_1[1].set_ylabel("Obsfucation Start\nSize 50 Samples", fontsize=13)

spans = []
lines = []
def hover(event):
    if event.button == 1:
        for span in spans:
            try:
                span.remove()
            except:
                pass
        if len(axs_1[0].lines) > 1:
            axs_1[0].lines.pop()
        plt.draw()
        spans.append(axs_1[0].axvspan(event.xdata-offset, event.xdata+299, facecolor='0.5', alpha=0.4))
        spans.append(axs_1[0].axvspan(event.xdata-offset + event.ydata, event.xdata-offset + event.ydata + 50, facecolor='b', alpha=0.20))
        spans.append(axs_1[0].axvspan(event.xdata-2, event.xdata+2, facecolor='r', alpha=1))
        lines.append(axs_1[0].plot(np.arange(start_index,end_index-50), obs_row[int(event.ydata)]))
    plt.draw() #redraw

fig_1.canvas.mpl_connect("motion_notify_event", hover)
plt.show()