import json
import os
from datetime import datetime
from typing import Tuple

import torch
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch import nn as nn

import common.evaluation.metrics as metrics
import common.utils.module as module_utils
from common.evaluation.evaluators import SupervisedTrainEvaluator
from common.evaluation.evaluators import TrainEvaluator, Evaluator
from common.train.callbacks import Callback, TerminateOnNaN
from common.train.callbacks import Checkpoint, MetricsPlotter, ComposeCallback, FileProgressLogger, ConsoleProgressLogger, StopOnZeroTrainLoss
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from common.train.tuning import ConfigurableTrainerFitResult
from common.train.tuning.configurable_trainer_base import ConfigurableTrainerBase
from datasets.matrix_sensing_dataset import MatrixSensingDataset
from evaluators.dln_matrix_evaluator import DLNMatrixValidationEvaluator
from models.dln_model_factory import DLNModelFactory
from trainers.dln_matrix_sensing_trainer import DLNMatrixSensingTrainer


class DLNMatrixSensingConfigurableTrainer(ConfigurableTrainerBase):

    def initialize(self, params: dict, state: dict):
        super().initialize(params, state)

        now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        output_dir = params["output_dir"]
        experiment_name = params["experiment_name"]
        state["experiment_dir"] = os.path.join(output_dir, f"{experiment_name}_{now_utc_str}")
        os.makedirs(state["experiment_dir"])

        matrix_sensing_dataset = MatrixSensingDataset.load(params["dataset_path"])
        num_samples = params["num_samples"]
        state["dataset"] = MatrixSensingDataset(matrix_sensing_dataset.A[:num_samples], matrix_sensing_dataset.y[:num_samples],
                                                matrix_sensing_dataset.target_matrix)

    def create_model(self, params: dict, state: dict) -> nn.Module:
        target_matrix = state["dataset"].target_matrix
        return DLNModelFactory.create_same_dim_deep_linear_network(target_matrix.size(0), target_matrix.size(1), depth=params["depth"],
                                                                   weight_init_type=params["weight_init_type"], alpha=params["init_alpha"],
                                                                   use_balanced_init=params["use_balanced_init"],
                                                                   enforce_pos_det=params["enforce_pos_det"],
                                                                   enforce_neg_det=params["enforce_neg_det"])

    def create_train_dataloader(self, params: dict, state: dict) -> torch.utils.data.DataLoader:
        train_dataset = state["dataset"]
        return torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    def create_train_and_validation_evaluators(self, model: nn.Module, device, params: dict, state: dict) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = [metrics.MetricInfo("mse_loss", metrics.MSELoss()),
                                 metrics.MetricInfo("sse_loss_div_2", metrics.MSELoss(reduction="sum", normalization_const=2))]
        train_evaluator = SupervisedTrainEvaluator(train_metric_info_seq)

        tracked_e2e_value_indices = params["tracked_e2e_value_of"]
        target_matrix = state["dataset"].target_matrix
        val_evaluator = DLNMatrixValidationEvaluator(model, target_matrix, tracked_e2e_value_indices=tracked_e2e_value_indices,
                                                     track_singular_values=params["track_singular_values"], device=state["device"])
        return train_evaluator, val_evaluator

    def create_trainer_callback(self, model: nn.Module, params: dict, state: dict) -> Callback:
        additional_metadata = {
            "Number of model parameters": module_utils.get_number_of_parameters(model),
        }

        logging_callback = self.__class__.__create_logging_callback(params, state, additional_metadata=additional_metadata)
        callbacks_list = [logging_callback]

        if params["plot_metrics"]:
            callbacks_list.append(MetricsPlotter(state["experiment_dir"], create_plots_interval=params["save_every_num_val"] * params["validate_every"]))

        if params["store_checkpoints"]:
            callbacks_list.append(Checkpoint(state["experiment_dir"], save_interval=params["save_every_num_val"] * params["validate_every"], n_saved=1))

        train_loss_fn = lambda trainer: trainer.train_evaluator.get_tracked_values()["sse_loss_div_2"].current_value
        callbacks_list.append(StopOnZeroTrainLoss(train_loss_fn=train_loss_fn, tol=params["stop_on_zero_loss_tol"]))
        callbacks_list.append(TerminateOnNaN(verify_batches=False))

        return ComposeCallback(callbacks_list)

    @staticmethod
    def __create_logging_callback(params, state, additional_metadata=None):
        if params["file_log"]:
            return FileProgressLogger(state["experiment_dir"], experiment_name=params["experiment_name"], train_batch_log_interval=-1,
                                      epoch_log_interval=params["validate_every"], run_params=params, additional_metadata=additional_metadata)
        return ConsoleProgressLogger(train_batch_log_interval=-1, epoch_log_interval=params["validate_every"],
                                     run_params=params, additional_metadata=additional_metadata)

    def create_trainer(self, model: nn.Module, train_evaluator: TrainEvaluator, val_evaluator: Evaluator, callback: Callback, device, params: dict,
                       state: dict) -> Trainer:
        optimizer = optim.SGD(model.parameters(), lr=params["lr"])
        mse_loss = nn.MSELoss()
        loss = lambda y_pred, y: mse_loss(y_pred, y) * (len(y_pred) / 2)
        return DLNMatrixSensingTrainer(model, optimizer, loss, train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                       callback=callback, device=device)

    def create_fit_result(self, trainer: Trainer, fit_output: FitOutput, params: dict, state: dict) -> ConfigurableTrainerFitResult:
        sse_loss_div_2 = fit_output.train_tracked_values["sse_loss_div_2"]
        score = sse_loss_div_2.current_value if sse_loss_div_2.current_value is not None else -1
        score_epoch = sse_loss_div_2.epochs_with_values[-1] if len(sse_loss_div_2.epochs_with_values) > 0 else -1

        if params["save_results"]:
            self.__save_results(trainer, fit_output, state)

        return ConfigurableTrainerFitResult(score, "sse_loss_div_2", score_epoch=score_epoch)

    def __save_results(self, trainer: Trainer, fit_output: FitOutput, state):
        if not os.path.exists(state["experiment_dir"]):
            os.makedirs(state["experiment_dir"])

        torch.save(trainer.model.compute_prod_matrix().detach(), os.path.join(state["experiment_dir"], "final_e2e_matrix.pt"))

        metric_values = self.__get_metric_values_from_fit_output(fit_output, trainer)
        with open(os.path.join(state["experiment_dir"], "final_metrics_values.json"), "w") as f:
            json.dump(metric_values, f, indent=2)

    def __get_metric_values_from_fit_output(self, fit_output: FitOutput, trainer: Trainer):
        metric_values = {}
        metric_values.update({name: tracked_value.current_value for name, tracked_value in fit_output.train_tracked_values.items()})
        metric_values.update({name: tracked_value.current_value for name, tracked_value in fit_output.val_tracked_values.items()})
        metric_values.update({name: tracked_value.current_value for name, tracked_value in fit_output.value_store.tracked_values.items()})
        metric_values.update({name: value for name, value in fit_output.value_store.other_values.items()})
        metric_values["fit_epochs"] = trainer.epoch + 1
        return metric_values
