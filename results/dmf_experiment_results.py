import glob
import json

import torch

import common.utils.logging as logging_utils


class DeepMatrixFactorizationExperimentResults:

    def __init__(self, experiment_dir, model_dir_prefix="model_", logger=None):
        self.experiment_dir = experiment_dir
        self.model_dir_prefix = model_dir_prefix
        self.logger = logger if logger is not None else logging_utils.get_default_logger()

    def load_final_e2e_matrices(self, verbose=True):
        e2e_matrices = []

        e2e_matrix_file_paths = glob.glob(f"{self.experiment_dir}/{self.model_dir_prefix}*/final_e2e_matrix.pt")
        if verbose:
            self.logger.info(f"Found {len(e2e_matrix_file_paths)} final e2e matrix files in experiment dir {self.experiment_dir}.")

        for path in sorted(e2e_matrix_file_paths):
            e2e_matrices.append(torch.load(path, map_location=torch.device("cpu")).detach())

        return e2e_matrices

    def load_final_metric_values_seq(self, verbose=True):
        final_metric_values_seq = []

        final_metric_values_file_paths = glob.glob(f"{self.experiment_dir}/{self.model_dir_prefix}*/final_metrics_values.json")
        if verbose:
            self.logger.info(f"Found {len(final_metric_values_file_paths)} final metrics values files in experiment dir {self.experiment_dir}.")

        for path in sorted(final_metric_values_file_paths):
            with open(path) as f:
                final_metric_values_seq.append(json.load(f))

        return final_metric_values_seq

    def load_min_nuc_norm_sols(self, verbose=True):
        min_nuc_norm_sols = []

        min_nuc_norm_sols_file_path = glob.glob(f"{self.experiment_dir}/{self.model_dir_prefix}*/min_nuc_norm_matrix.pt")
        if verbose:
            self.logger.info(f"Found {len(min_nuc_norm_sols_file_path)} min nuclear norm solution files in experiment dir {self.experiment_dir}.")

        for path in sorted(min_nuc_norm_sols_file_path):
            min_nuc_norm_sols.append(torch.load(path, map_location=torch.device("cpu")).detach())

        return min_nuc_norm_sols
