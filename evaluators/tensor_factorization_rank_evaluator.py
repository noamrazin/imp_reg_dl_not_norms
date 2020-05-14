from typing import Dict

import torch

import common.utils.module as module_utils
import evaluators.tensor_rank_metrics as trm
from common.evaluation.evaluators import Evaluator, MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo
from common.train.tracked_value import TrackedValue
from models.tensor_cp_factorization import TensorCPFactorization


class TensorFactorizationCPRankEvaluator(Evaluator):

    def __init__(self, model: TensorCPFactorization, tol=1e-6, device=module_utils.get_device()):
        self.model = model

        self.model = model
        self.tol = tol
        self.device = device
        self.metric_infos = self.__create_tensor_rank_metric_infos()
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_tensor_rank_metric_infos(self):
        metric_infos = {
            "model_tensor_rank": MetricInfo("model_tensor_rank", trm.ModelTensorCPRank(tol=self.tol))
        }

        return metric_infos

    def get_metric_infos(self) -> Dict[str, MetricInfo]:
        return self.metric_infos

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics

    def get_tracked_values(self) -> Dict[str, TrackedValue]:
        return self.tracked_values

    def evaluate(self) -> dict:
        with torch.no_grad():
            self.model.to(self.device)
            model_tensor = self.model.compute_tensor()

            metric_values = {}
            for name, metric in self.metrics.items():
                value = metric(model_tensor)

                if value != -1:
                    self.tracked_values[name].add_batch_value(value)

                metric_values[name] = value

            return metric_values
