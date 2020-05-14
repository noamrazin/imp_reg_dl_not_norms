import json
import os
import re

import torch


class TensorCompletionExperimentResults:

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.experiment_done = os.path.exists(os.path.join(self.experiment_dir, "tensor_rank_and_reconstruction_metrics.json"))

        if self.experiment_done:
            self.num_samples_regex = re.compile(r".*samples_(\d+)_.*")
            self.init_std_regex = re.compile(r".*init_(\d+[^_]*)_.*")

            self.model_tensor = torch.load(os.path.join(self.experiment_dir, "model_tensor.pt"))
            self.memorizer_tensor = torch.load(os.path.join(self.experiment_dir, "memorizer_tensor.pt"))
            self.target_tensor = torch.load(os.path.join(self.experiment_dir, "targets_tensor.pt"))

            with open(os.path.join(self.experiment_dir, "tensor_rank_and_reconstruction_metrics.json")) as f:
                self.tensor_rank_and_reconstruction_metric_values = json.load(f)

    def get_num_samples(self):
        match = self.num_samples_regex.match(self.experiment_dir)
        return int(match.group(1))

    def get_init_std_str(self):
        match = self.init_std_regex.match(self.experiment_dir)
        if not match:
            return ""

        init_std_str = match.group(1)
        return init_std_str

    def get_model_tensor(self):
        return self.model_tensor

    def get_memorizer_tensor(self):
        return self.memorizer_tensor

    def get_target_tensor(self):
        return self.target_tensor

    def get_tensor_rank_and_reconstruction_metric_values(self):
        return self.tensor_rank_and_reconstruction_metric_values
