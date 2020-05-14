from typing import Dict
from typing import Sequence

import torch

import common.utils.module as module_utils
from common.evaluation.evaluators import Evaluator, MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo, DummyAveragedMetric
from common.train.tracked_value import TrackedValue
from evaluators.tensor_metrics import MatrixNormMetric, MatrixEffectiveRankMetric, ReconstructionErrorMetric, MatrixRankMetric, MatrixEntryMetric
from models.deep_linear_net import DeepLinearNet


class DLNMatrixValidationEvaluator(Evaluator):
    SINGULAR_VALUE_METRIC_NAME_TEMPLATE = "singular_value_{0}"

    def __init__(self, model: DeepLinearNet, target_matrix, norms: Sequence[str] = ("fro", "nuc"), track_rank: bool = True,
                 track_reconstruction_metrics: bool = False, tracked_e2e_value_indices=None, track_singular_values: bool = True,
                 device=module_utils.get_device()):
        self.model = model
        self.target_matrix = target_matrix
        self.norms = norms
        self.track_rank = track_rank
        self.track_reconstruction_metrics = track_reconstruction_metrics
        self.tracked_e2e_value_indices = tracked_e2e_value_indices if tracked_e2e_value_indices is not None else []
        self.track_singular_values = track_singular_values
        self.device = device

        self.matrix_metric_infos = self.__create_norms_metric_infos(norms, track_rank)
        self.matrix_metric_infos.update(self.__create_matrix_values_metric_infos(self.tracked_e2e_value_indices))
        self.matrix_metrics = {name: metric_info.metric for name, metric_info in self.matrix_metric_infos.items()}

        self.combined_metric_infos = {}
        self.combined_metric_infos.update(self.matrix_metric_infos)

        if track_reconstruction_metrics:
            self.reconstruction_metric_infos = self.__create_reconstruction_metric_infos()
            self.reconstruction_metrics = {name: metric_info.metric for name, metric_info in self.reconstruction_metric_infos.items()}
            self.combined_metric_infos.update(self.reconstruction_metric_infos)

        if self.track_singular_values:
            self.singular_values_metric_infos = self.__create_singular_values_metric_infos()
            self.singular_values_metrics = {name: metric_info.metric for name, metric_info in self.singular_values_metric_infos.items()}
            self.combined_metric_infos.update(self.singular_values_metric_infos)

        self.combined_metrics = {name: metric_info.metric for name, metric_info in self.combined_metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.combined_metric_infos)

    def __create_norms_metric_infos(self, norms, track_rank):
        metric_infos = {}
        for norm in norms:
            metric_name = f"{norm}_e2e_norm"
            metric = MatrixNormMetric(norm)
            metric_info = MetricInfo(metric_name, metric)

            metric_infos[metric_name] = metric_info

        if track_rank:
            metric_name = "e2e_effective_rank"
            metric = MatrixEffectiveRankMetric()
            metric_info = MetricInfo(metric_name, metric)

            metric_infos[metric_name] = metric_info

            metric_name = "e2e_rank"
            metric = MatrixRankMetric()
            metric_info = MetricInfo(metric_name, metric)

            metric_infos[metric_name] = metric_info

        return metric_infos

    def __create_matrix_values_metric_infos(self, tracked_e2e_value_indices, use_same_plot=True):
        metric_infos = {}
        for i, j in tracked_e2e_value_indices:
            metric_name = f"e2e_{i}_{j}_value"
            metric = MatrixEntryMetric(i, j)
            metric_tag = "E2E Matrix Values" if use_same_plot else ""
            metric_infos[metric_name] = MetricInfo(metric_name, metric, tag=metric_tag)

        return metric_infos

    def __create_reconstruction_metric_infos(self):
        return {
            "reconstruction_error": MetricInfo("reconstruction_error", ReconstructionErrorMetric()),
            "normalized_reconstruction_error": MetricInfo("normalized_reconstruction_error", ReconstructionErrorMetric(normalized=True))
        }

    def __create_singular_values_metric_infos(self, use_same_plot=True):
        metric_infos = {}

        for i in range(min(self.target_matrix.size(0), self.target_matrix.size(1))):
            singular_value_metric_name = self.SINGULAR_VALUE_METRIC_NAME_TEMPLATE.format(i)
            metric_tag = "E2E Singular Values" if use_same_plot else ""
            metric_infos[singular_value_metric_name] = MetricInfo(singular_value_metric_name, DummyAveragedMetric(), tag=metric_tag)

        return metric_infos

    def get_metric_infos(self) -> Dict[str, MetricInfo]:
        return self.combined_metric_infos

    def get_metrics(self) -> Dict[str, Metric]:
        return self.combined_metrics

    def get_tracked_values(self) -> Dict[str, TrackedValue]:
        return self.tracked_values

    def evaluate(self) -> dict:
        with torch.no_grad():
            self.model.to(self.device)

            end_to_end_matrix = self.model.compute_prod_matrix()
            end_to_end_matrix = end_to_end_matrix.to(self.device)
            target_matrix = self.target_matrix.to(self.device)

            metric_values = {}

            for name, metric in self.matrix_metrics.items():
                value = metric(end_to_end_matrix)
                self.tracked_values[name].add_batch_value(value)
                metric_values[name] = value

            if self.track_reconstruction_metrics:
                for name, metric in self.reconstruction_metrics.items():
                    value = metric(end_to_end_matrix, target_matrix)
                    self.tracked_values[name].add_batch_value(value)
                    metric_values[name] = value

            if self.track_singular_values:
                svd_result = torch.svd(end_to_end_matrix, compute_uv=False)
                singular_values = svd_result.S
                for i in range(min(end_to_end_matrix.size(0), end_to_end_matrix.size(1))):
                    singular_value_metric_name = self.SINGULAR_VALUE_METRIC_NAME_TEMPLATE.format(i)
                    value = singular_values[i].item()
                    self.singular_values_metrics[singular_value_metric_name](value)
                    self.tracked_values[singular_value_metric_name].add_batch_value(value)
                    metric_values[singular_value_metric_name] = value

            return metric_values
