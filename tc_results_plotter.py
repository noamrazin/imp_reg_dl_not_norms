import argparse
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from results.tensor_completion_experiment_results import TensorCompletionExperimentResults


class MetricPlotInfo:

    def __init__(self, xlabel, ylabel):
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.name_to_x_values = defaultdict(list)
        self.name_to_values = defaultdict(list)
        self.name_to_errs = defaultdict(list)

        self.ground_truth_value = None
        self.colors = list(plt.get_cmap("tab10").colors)
        self.colors.pop(3)  # Remove red color since it is used for ground truth
        self.colors.pop(3)  # Remove purple
        self.markers = ["o", "v", "^", "D", "s", "X", "h", "1"]
        self.linestyles = ["--", "-", "-."]


def __create_model_to_per_sample_size_metrics_dict(relevant_experiments_dir):
    dirs_of_experiments = glob.glob(f"{relevant_experiments_dir}/c_tensor_rank*")

    print(f"Found {len(dirs_of_experiments)} experiments in experiments dir {relevant_experiments_dir}.")

    model_to_per_sample_size_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for experiment_dir in dirs_of_experiments:
        experiment_results = TensorCompletionExperimentResults(experiment_dir)
        if not experiment_results.experiment_done:
            continue

        metric_values = experiment_results.get_tensor_rank_and_reconstruction_metric_values()
        sample_size = experiment_results.get_num_samples()

        init_std_str = experiment_results.get_init_std_str()

        __add_memorizer_metrics_for_num_samples(model_to_per_sample_size_metrics, sample_size, metric_values)
        __add_targets_metrics_for_num_samples(model_to_per_sample_size_metrics, sample_size, metric_values)

        sample_to_metrics_dict = model_to_per_sample_size_metrics[__get_model_name(init_std_str)]
        metrics = sample_to_metrics_dict[sample_size]

        metrics["reconstruction error"].append(metric_values["model_reconstruction_error"])
        metrics["rank"].append(metric_values["model_tensor_rank"])

    return model_to_per_sample_size_metrics


def __get_model_name(init_std_str):
    if init_std_str != "":
        return f"tf, init {init_std_str}"

    return "model"


def __add_memorizer_metrics_for_num_samples(model_to_per_sample_size_metrics, sample_size, metric_values):
    memorizer_metrics_for_sample_size = model_to_per_sample_size_metrics["linear"][sample_size]
    if len(memorizer_metrics_for_sample_size) != 0:
        # Memorizer metrics for this sample size were already added
        return

    memorizer_metrics_for_sample_size["reconstruction error"].append(metric_values["memorizer_reconstruction_error"])
    memorizer_metrics_for_sample_size["rank"].append(metric_values["memorizer_tensor_rank"])


def __add_targets_metrics_for_num_samples(model_to_per_sample_size_metrics, sample_size, metric_values):
    # The targets metrics should be the same for each sample size, this is just allows to populate the metric for each sample size for plotting.
    targets_metrics_for_sample_size = model_to_per_sample_size_metrics["ground truth"][sample_size]
    if len(targets_metrics_for_sample_size) != 0:
        # Targets metrics for this sample size were already added
        return

    targets_metrics_for_sample_size["rank"].append(metric_values["targets_tensor_rank"])


def __create_metric_plot_infos(model_to_per_sample_size_metrics):
    metric_name_to_plot_info = {}

    for model, per_sample_size_metrics in model_to_per_sample_size_metrics.items():
        for sample_size in sorted(list(per_sample_size_metrics.keys())):
            metrics = per_sample_size_metrics[sample_size]

            for metric_name, metric_values in metrics.items():
                if metric_name not in metric_name_to_plot_info:
                    metric_name_to_plot_info[metric_name] = MetricPlotInfo(xlabel="# of observations", ylabel=metric_name)

                metric_plot_info = metric_name_to_plot_info[metric_name]

                if model == "ground truth" and metric_plot_info.ground_truth_value is None:
                    metric_plot_info.ground_truth_value = metric_values[0]

                median = np.median(metric_values).item()
                upper_err = np.percentile(metric_values, q=75).item() - median
                lower_err = median - np.percentile(metric_values, q=25).item()

                metric_plot_info.name_to_x_values[model].append(sample_size)
                metric_plot_info.name_to_values[model].append(median)
                metric_plot_info.name_to_errs[model].append([lower_err, upper_err])

    return metric_name_to_plot_info


def populate_plot_for_metric(ax, metric_plot_info, exclude_models):
    ax.set_ylabel(metric_plot_info.ylabel, fontsize=14)
    ax.set_xlabel(metric_plot_info.xlabel, fontsize=14)

    min_samples = float('inf')
    not_excluded_models = sorted([name for name in metric_plot_info.name_to_values.keys() if name not in exclude_models])

    for i, name in enumerate(not_excluded_models):
        y_values = metric_plot_info.name_to_values[name]
        y_errs = metric_plot_info.name_to_errs[name]
        y_errs = np.array([[err[0] for err in y_errs], [err[1] for err in y_errs]])
        x_values = metric_plot_info.name_to_x_values[name]

        min_samples = min(min_samples, x_values[0])
        color_index = i % len(metric_plot_info.colors)
        marker_index = i % len(metric_plot_info.markers)
        label = name

        # Plots twice to create transparent error bars
        ax.plot(x_values, y_values, label=label, marker=metric_plot_info.markers[marker_index],
                color=metric_plot_info.colors[color_index], linestyle=metric_plot_info.linestyles[i % len(metric_plot_info.linestyles)])
        ax.errorbar(x_values, y_values, yerr=y_errs, alpha=0.6, marker=metric_plot_info.markers[marker_index],
                    color=metric_plot_info.colors[color_index],
                    linestyle=metric_plot_info.linestyles[i % len(metric_plot_info.linestyles)])

    if metric_plot_info.ground_truth_value is not None:
        ax.axhline(metric_plot_info.ground_truth_value, label="ground truth", color="C3", linewidth=3.0)
        ax.set_ylim(bottom=0)
        ax.legend()
    else:
        ax.set_yscale("log")

    ax.tick_params(labelsize=10)
    ax.autoscale(enable=True, axis='x', tight=True)


def create_tensor_completion_plots(experiments_dir, plot_title, exclude_models=None, save_plot_to=""):
    exclude_models = exclude_models if exclude_models is not None else []
    fig, axes = plt.subplots(1, 2, figsize=(5.63, 2.95))

    model_to_per_sample_size_metrics = __create_model_to_per_sample_size_metrics_dict(experiments_dir)
    metric_name_to_plot_info = __create_metric_plot_infos(model_to_per_sample_size_metrics)

    for j, metric_plot_info in enumerate(metric_name_to_plot_info.values()):
        ax = axes[j]
        populate_plot_for_metric(ax, metric_plot_info, exclude_models)

    plt.suptitle(plot_title, fontsize=15, x=0.55)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.32)
    if save_plot_to:
        plt.savefig(save_plot_to, dpi=250, bbox_inches='tight', pad_inches=0.01)

    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-experiments_dir", type=str, required=True, help="Paths to a directory with experiments.")
    p.add_argument("-plot_title", type=str, default="", help="Title for the plot.")
    p.add_argument("-save_plot_to", type=str, default="", help="Save plot to the given file path (doesn't save if none given).")
    args = p.parse_args()

    exclude_models = ["ground truth"]
    create_tensor_completion_plots(args.experiments_dir, args.plot_title, exclude_models, args.save_plot_to)


if __name__ == "__main__":
    main()
