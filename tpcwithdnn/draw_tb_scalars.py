"""
Plot TensorBoard comparison of models learning curves,
with a prettier formatting than the TensorBoard's one.
"""
import os
import datetime
import re

import numpy as np
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib import cm

def plot_scalar():
    csv_path = "sel-logs-csv"
    file_formats = ["png"]
    regex = re.compile("^run-phi([0-9]+)_r([0-9]+)_z([0-9]+)_" +\
                       "nEv([0-9]+)_([a-z]+)-tag-epoch_root_mean_squared_error.csv$")
    csv_files = list(sorted(filter(regex.match, os.listdir(csv_path))))

    rc("font",**{"family":"sans-serif","sans-serif":["Helvetica"], "size": 18})
    cmap = cm.get_cmap("tab20")
    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes()
    for ind, csv_file in enumerate(csv_files):
        file_search = re.search(regex, csv_file)
        train_or_val = file_search.group(5)
        label = "%s, %s x %s x %s, %s" % (file_search.group(4), file_search.group(1),
                file_search.group(2), file_search.group(3), train_or_val)
        data = np.genfromtxt("%s/%s" % (csv_path, csv_file), delimiter=",")[1:, 1:]
        ax.plot(data[:, 0], data[:, 1], label=label, c=cmap(ind / 20))

    ax.set_xlabel("Epochs")
    #ax.set_ylim([0., 0.012])
    #ax.set_xlim([-1., 21])
    ax.set_yticks(np.arange(0, 0.25, 0.02))
    ax.set_xticks(np.arange(0, 22, 2))
    ax.set_ylabel(r"$\it{RMSE}$")
    leg_title = r"Train setup: $\it{N}_{ev}^{training}, \it{n}_{\it{\varphi}}$" +\
                r"$ \times \it{n}_{\it{r}} \times \it{n}_{\it{z}}$"
    ax.legend(loc="upper right", title=leg_title)
    ax.grid(which="major")

    fig.tight_layout()
    date = datetime.date.today().strftime("%Y%m%d")
    out_filename = "%s_tb_scalar" % date
    for ff in file_formats:
        fig.savefig("%s.%s" % (out_filename, ff), bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    plot_scalar()
