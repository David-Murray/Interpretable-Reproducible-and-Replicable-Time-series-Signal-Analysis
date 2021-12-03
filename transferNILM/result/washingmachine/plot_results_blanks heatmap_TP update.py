import os
import sys
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
importance = np.loadtxt(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../importance.csv')))
fig_1, axs_1 = plt.subplots(3, figsize=(15,9), sharex=True)
fig_1.suptitle('Learned Features via Obfuscation')
fig_1.subplots_adjust(wspace=0, hspace=0)


flag_x = False
centre_index = 705900 # 8585 # 8500 # 959750 # 152350
plot_size = 599
alpha_val = 1
blank_size = 50
obs_row = None
diag_rows = None
start_index = centre_index - offset
end_index = start_index + plot_size 
axs_1[0].plot(np.arange(0,plot_size), mains_dict[f"run_0"][start_index+299-12:start_index+299+plot_size-12], alpha=1, label="Aggregate")

axs_1[0].plot(np.arange(0,end_index-start_index),target_dict[f"run_0"][start_index-12:end_index-12], alpha=1, label="Target")
axs_1[0].plot(np.arange(0,end_index-start_index),prediction_none[start_index-12:end_index-12], alpha=1, label="Prediction_Static")
axs_1[0].plot(np.arange(0,end_index-start_index),prediction_none[start_index-12:end_index-12], alpha=1, label="Prediction")

divider0 = make_axes_locatable(axs_1[0])
cax0 = divider0.append_axes("right", size="5%", pad=.05)
cax0.remove()

axs_1[1].plot(np.arange(0,end_index-start_index),importance[:599], alpha=1, label="Gradients")
axs_1[1].set_ylabel("Cumulative\nIntegrated\nGradients", fontsize=13)
divider1 = make_axes_locatable(axs_1[1])
cax1 = divider1.append_axes("right", size="5%", pad=.05)
cax1.remove()

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
    new_stack = np.array(prediction_dict[f"run_{i}"][start_index:end_index-blank_size])# - prediction_none[start_index:end_index-blank_size]
    if obs_row is None:
        obs_row = new_stack
    else:
        obs_row = np.vstack((obs_row, new_stack))
obs_row_blanks = np.fliplr(np.triu(np.fliplr(obs_row), k=-330))
obs_row_blanks = np.fliplr(np.tril(np.fliplr(obs_row_blanks), k=300))

axs_1[0].set_title(f"Obfuscation (Test Index: {centre_index})", fontsize=13)
# axs_1[0].set_xlabel("Sample", fontsize=13)
axs_1[0].set_ylabel("Prediction\n(Watts)", fontsize=13)
axs_1[0].legend(loc='upper right')
# axs_1[0].set_xlim(0, end_index-start_index) # minus 20 samples lines up line plot with heatmap

imshow = axs_1[2].imshow(obs_row,  aspect='auto', cmap='nipy_spectral', extent=[0,599,550,0])
imshow_blanks = axs_1[2].imshow(obs_row_blanks,  aspect='auto', cmap='nipy_spectral', extent=[0,599,550,0])
divider2 = make_axes_locatable(axs_1[2])
cax2 = divider2.append_axes("right", size="5%", pad=.05)
# divider = make_axes_locatable(axs_1[2])
# cax = divider.append_axes("right", size="3%", pad=0.05)
cbar = fig_1.colorbar(imshow, cax=cax2)
cbar.set_label("Power (Watts)", fontsize=12)

# Feature Area 1
axs_1[0].axvspan(705666-start_index, 705666+50-start_index, facecolor='0.5', alpha=0.25)
axs_1[0].annotate('1.', xy=(705666+5-start_index, 2000), xycoords='data')
axs_1[2].add_patch(Ellipse((705730-start_index, 222),
        width=70,
        height=40,
        angle=-60,
        linewidth=1, fill=False))
axs_1[2].annotate('1.', xy=(705705-start_index, 200), xycoords='data', fontsize=12)
# Feature Area 2
axs_1[0].axvspan(705929-start_index, 705929+50-start_index, facecolor='0.5', alpha=0.25)
axs_1[0].annotate('2.', xy=(705929+5-start_index, 2000), xycoords='data')
axs_1[2].add_patch(Ellipse((705737-start_index, 482),
        width=45,
        height=35,
        angle=-60,
        linewidth=1, fill=False))
axs_1[2].annotate('2.', xy=(705735-start_index, 488), xycoords='data', fontsize=12)
# Feature Area 3
axs_1[0].axvspan(705831-start_index, 705850+75-start_index, facecolor='0.5', alpha=0.25)
axs_1[0].annotate('3.', xy=(705850+5-start_index, 2000), xycoords='data')
axs_1[2].add_patch(Ellipse((705710-start_index, 440),
        width=80,
        height=45,
        angle=-75,
        linewidth=1, fill=False))
axs_1[2].annotate('3.', xy=(705701-start_index, 445), xycoords='data', fontsize=12)


axs_1[2].set_xlabel("Sample", fontsize=13)
axs_1[2].set_ylabel("Obsfucation\nStart Index", fontsize=13)

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
        lines.append(axs_1[0].plot(np.arange(start_index-start_index,end_index-start_index-50), obs_row[int(event.ydata)]))
    plt.draw() #redraw

fig_1.canvas.mpl_connect("motion_notify_event", hover)
plt.show()