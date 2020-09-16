import argparse

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from matplotlib.collections import LineCollection

MARKERS = ["o", "v", "^", "D", "s", "1", "^", "h"]


def create_training_dynamics_plots(checkpoints, experiment_name, plot_title="", min_loss=1e-4, save_plot_to=""):
    train_evaluators_tracked_values = [checkpoint["train_evaluator"] for checkpoint in checkpoints]
    val_evaluators_tracked_values = [checkpoint["val_evaluator"] for checkpoint in checkpoints]

    fig, ax = plt.subplots(1, figsize=(3.1, 3.4))

    __populate_entry_to_loss_plot(ax, train_evaluators_tracked_values, val_evaluators_tracked_values, experiment_name, plot_title, min_loss)

    plt.tight_layout()
    if save_plot_to:
        plt.savefig(save_plot_to, dpi=250, bbox_inches='tight', pad_inches=0.01)

    plt.show()


def __populate_entry_to_loss_plot(ax, train_evaluators_tracked_values, val_evaluators_tracked_values, experiment_names, title="", min_loss=1e-4):
    loss_values_seq, unobserved_entry_abs_values_seq, iterations_seq = __extract_plot_info_from_evaluators(train_evaluators_tracked_values,
                                                                                                           val_evaluators_tracked_values,
                                                                                                           min_loss)

    max_iter = max([iterations[-1] for iterations in iterations_seq])
    norm = mpl.colors.Normalize(vmin=1, vmax=max_iter)
    adjusted_cmap = cmap_map(copper_to_blue_shades_transform, matplotlib.cm.copper_r)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=adjusted_cmap)
    color_bar_ticks = np.linspace(1, max_iter, num=4)
    cb = plt.colorbar(cmap, ticks=color_bar_ticks, format=ColorBarIterationsFormatter(), orientation="horizontal", pad=0.19)
    cb.set_label(label="iterations", size=12)
    cb.ax.tick_params(labelsize=9)

    for i, (loss_values, values, iterations) in enumerate(zip(loss_values_seq, unobserved_entry_abs_values_seq, iterations_seq)):
        points = np.array([loss_values, values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=adjusted_cmap, norm=norm, linewidths=2)
        lc.set_array(iterations)
        ax.add_collection(lc)
        # Adds markers over plots
        marker_places = __create_marker_places(loss_values, min_loss)
        ax.plot(loss_values, values, label=experiment_names[i], linestyle="none", color=plt.get_cmap("gray_r")(0.35 + i * 0.17),
                marker=MARKERS[i % len(MARKERS)], markevery=marker_places)

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel("|unobserved entry|", fontsize=13)
    ax.set_xlabel("loss", fontsize=13)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xlim(right=min_loss)
    ax.autoscale(enable=True, axis='y')
    ax.set_ylim(bottom=0)
    ax.legend(handletextpad=0.15, handlelength=1.13, prop={'size': 9})


def __create_marker_places(loss_values, min_loss):
    marker_loss_values = []
    j = 1
    while 10 ** -j >= min_loss:
        marker_loss_values.append(1.3 * 10 ** - j)
        j += 1

    marker_places = []
    curr = 0
    for j, loss_value in enumerate(loss_values):
        if loss_value < marker_loss_values[curr]:
            marker_places.append(j)
            curr += 1

        if curr == len(marker_loss_values):
            break

    return marker_places


def __extract_plot_info_from_evaluators(train_evaluators_tracked_values, val_evaluators_tracked_values, min_loss):
    loss_values_seq, unobserved_entry_abs_values_seq, iterations_seq = [], [], []
    for i, (train_tracked_values, val_tracked_values) in enumerate(zip(train_evaluators_tracked_values, val_evaluators_tracked_values)):
        loss_tracked_value = train_tracked_values["sse_loss_div_2"]
        unobserved_entry_tracked_value = val_tracked_values["e2e_0_0_value"]

        epochs_with_unobserved_entry_value = unobserved_entry_tracked_value["epochs_with_values"]
        loss_values = loss_tracked_value["epoch_values"]
        loss_values = [loss_values[i] for i in epochs_with_unobserved_entry_value]
        loss_values = __truncate_after_min_loss(loss_values, min_loss)

        values = unobserved_entry_tracked_value["epoch_values"][:len(loss_values)]
        values = [abs(value) for value in values]
        iterations = np.array(epochs_with_unobserved_entry_value[: len(values)])

        loss_values_seq.append(loss_values)
        unobserved_entry_abs_values_seq.append(values)
        iterations_seq.append(iterations)

    return loss_values_seq, unobserved_entry_abs_values_seq, iterations_seq


def __truncate_after_min_loss(loss_values, min_loss=1e-4):
    num_before_min = len(loss_values)
    for index, loss_value in enumerate(loss_values):
        if loss_value < min_loss:
            num_before_min = index + 1
            break

    return loss_values[:num_before_min]


class ColorBarIterationsFormatter(mticker.Formatter):

    def __call__(self, x, pos=None):
        str_x = str(int(x))
        return f"{str_x[0]}.{str_x[1]}e{len(str_x) - 1}" if len(str_x) > 1 else f"{str_x[0]}"


def copper_to_blue_shades_transform(x):
    new_x = x.copy()
    new_x[0] = x[2]
    new_x[2] = x[0]
    new_x = new_x * 0.85
    return new_x


# Adapted from https://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html
def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red', 'green', 'blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1],), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-experiments_checkpoint_paths", nargs="+", type=str, required=True, help="Paths to the experiments checkpoint.")
    p.add_argument("-experiments_names", nargs="+", type=str, required=True, help="Name for each experiment, appears as labels in the plot."
                                                                                  "Each checkpoint given must have a matching name.")
    p.add_argument("-plot_title", type=str, default="", help="Title for the plot.")
    p.add_argument("-min_loss", type=float, default=1e-4, help="Minimal loss value to plot until.")
    p.add_argument("-save_plot_to", type=str, default="", help="Save plot to the given file path (doesn't save if non given)")
    args = p.parse_args()

    if len(args.experiments_names) != len(args.experiments_checkpoint_paths):
        raise ValueError("Mismatch in number of experiments checkpoints and names given. Number of 'experiment_names' given should match that of "
                         "'experiments_checkpoint_path'.")

    checkpoints = [torch.load(checkpoint_path, map_location=torch.device("cpu")) for checkpoint_path in args.experiments_checkpoint_paths]
    create_training_dynamics_plots(checkpoints, args.experiments_names, args.plot_title, args.min_loss, args.save_plot_to)


if __name__ == "__main__":
    main()
