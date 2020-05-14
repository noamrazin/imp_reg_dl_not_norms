import json
import os
from datetime import datetime
from typing import Tuple

import torch
import torch.utils
import torch.utils.data
from torch import nn as nn

import evaluators.tensor_rank_metrics as trm
from common.evaluation import metrics as metrics
from common.evaluation.evaluators import SupervisedTrainEvaluator, SupervisedValidationEvaluator, TrainEvaluator, \
    Evaluator, ComposeEvaluator
from common.train import callbacks as callbacks
from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from common.train.trainers import SupervisedTrainer
from common.train.tuning import ConfigurableTrainerFitResult
from common.train.tuning.configurable_trainer_base import ConfigurableTrainerBase
from common.utils import module as module_utils
from datasets.tensor_completion_dataset import TensorCompletionDataset
from evaluators.tensor_factorization_rank_evaluator import TensorFactorizationCPRankEvaluator
from evaluators.tensor_metrics import ReconstructionErrorMetric, compute_reconstruction_error
from models.tensor_cp_factorization import TensorCPFactorization
from opt.group_rmsprop_optimizer import GroupRMSprop


class TensorCompletionConfigurableTrainer(ConfigurableTrainerBase):

    def initialize(self, params: dict, state: dict):
        super().initialize(params, state)

        now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        output_dir = params["output_dir"]
        experiment_name = params["experiment_name"]
        state["experiment_dir"] = os.path.join(output_dir, f"{experiment_name}_{now_utc_str}")
        os.makedirs(state["experiment_dir"])

        dataset = TensorCompletionDataset.load(params["dataset_path"])
        state["dataset"] = dataset

        train_indices = dataset.train_indices_order[:params["num_samples"]]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        state["train_dataset"] = train_dataset

    def create_model(self, params: dict, state: dict) -> nn.Module:
        dataset = state["dataset"]
        return TensorCPFactorization(num_dim_per_mode=list(dataset.target_tensor.size()), order=len(dataset.target_tensor.size()),
                                     init_std=params["init_std"])

    def create_train_dataloader(self, params: dict, state: dict) -> torch.utils.data.DataLoader:
        train_dataset = state["train_dataset"]
        return torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    def create_train_and_validation_evaluators(self, model: nn.Module, device, params: dict, state: dict) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = [metrics.MetricInfo("mse_loss", metrics.MSELoss())]
        train_evaluator = SupervisedTrainEvaluator(train_metric_info_seq)

        dataset = state["dataset"]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        reconstruction_error_metric_infos = [metrics.MetricInfo("reconstruction_error", ReconstructionErrorMetric(normalized=True))]
        reconstruction_evaluator = SupervisedValidationEvaluator(model, dataloader, reconstruction_error_metric_infos, device=device)

        val_evluators = [reconstruction_evaluator]
        if params["track_tensor_rank"]:
            val_evluators.append(TensorFactorizationCPRankEvaluator(model, tol=params["tensor_rank_tol"], device=device))

        return train_evaluator, ComposeEvaluator(val_evluators)

    def create_trainer_callback(self, model: nn.Module, params: dict, state: dict) -> Callback:
        additional_metadata = {
            "Number of model parameters": module_utils.get_number_of_parameters(model),
            "target_from_norm": torch.norm(state["dataset"].target_tensor, p="fro").item()
        }
        logging_callback = self.__class__.__create_logging_callback(params, state, additional_metadata=additional_metadata)

        callbacks_list = [logging_callback]

        if params["plot_metrics"]:
            callbacks_list.append(
                callbacks.MetricsPlotter(state["experiment_dir"], create_plots_interval=params["save_every_num_val"] * params["validate_every"]))

        if params["store_checkpoints"]:
            callbacks_list.append(
                callbacks.Checkpoint(state["experiment_dir"], save_interval=params["save_every_num_val"] * params["validate_every"], n_saved=1))

        train_loss_fn = lambda trainer: trainer.train_evaluator.get_tracked_values()["mse_loss"].current_value
        callbacks_list.append(callbacks.StopOnZeroTrainLoss(train_loss_fn=train_loss_fn, tol=params["stop_on_zero_loss_tol"], patience=2))
        callbacks_list.append(callbacks.TerminateOnNaN(verify_batches=False))

        return callbacks.ComposeCallback(callbacks_list)

    @staticmethod
    def __create_logging_callback(params, state, additional_metadata=None):
        if params["file_log"]:
            return callbacks.FileProgressLogger(state["experiment_dir"], experiment_name=params["experiment_name"],
                                                train_batch_log_interval=-1, epoch_log_interval=params["validate_every"],
                                                run_params=params, additional_metadata=additional_metadata)
        return callbacks.ConsoleProgressLogger(train_batch_log_interval=-1, epoch_log_interval=params["validate_every"], run_params=params,
                                               additional_metadata=additional_metadata)

    def create_trainer(self, model: nn.Module, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, params: dict,
                       state: dict) -> Trainer:
        if params["use_adaptive_lr"]:
            optimizer = GroupRMSprop(model.parameters(), lr=params["lr"])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])

        loss = nn.MSELoss()
        return SupervisedTrainer(model, optimizer, loss, train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                 callback=callback, device=device)

    def create_fit_result(self, trainer: Trainer, fit_output: FitOutput, params: dict,
                          state: dict) -> ConfigurableTrainerFitResult:
        dataset = state["dataset"]

        # Creates the "Memorizer" baseline predictions, that predict the given entries exactly and for the rest 0.
        train_indices = state["train_dataset"].indices
        memorizer_tensor = torch.zeros_like(dataset.target_tensor)
        train_indices_tensor = dataset.all_indices_tensor[train_indices]
        split_indices = [train_indices_tensor[:, i] for i in range(train_indices_tensor.size(1))]

        memorizer_tensor[split_indices] = dataset.target_tensor[split_indices]

        with torch.no_grad():
            model_tensor = trainer.model.compute_tensor()

        torch.save(model_tensor, os.path.join(state["experiment_dir"], "model_tensor.pt"))
        torch.save(memorizer_tensor, os.path.join(state["experiment_dir"], "memorizer_tensor.pt"))
        torch.save(dataset.target_tensor, os.path.join(state["experiment_dir"], "targets_tensor.pt"))

        additional_metadata = self.__create_tensor_rank_and_reconstruction_metrics(model_tensor, dataset.target_tensor,
                                                                                   memorizer_tensor, params)
        with open(os.path.join(state["experiment_dir"], "tensor_rank_and_reconstruction_metrics.json"), "w") as f:
            json.dump(additional_metadata, f, indent=2)

        mse_loss_tracked_value = fit_output.train_tracked_values["mse_loss"]
        score = mse_loss_tracked_value.current_value if mse_loss_tracked_value.current_value is not None else -1
        score_epoch = mse_loss_tracked_value.epochs_with_values[-1] if len(mse_loss_tracked_value.epochs_with_values) > 0 else -1
        return ConfigurableTrainerFitResult(score, "mse_loss",
                                            score_epoch=score_epoch,
                                            additional_metadata=additional_metadata)

    def __create_tensor_rank_and_reconstruction_metrics(self, model_tensor, targets_tensor, memorizer_tensor, params):
        metrics = {
            "model_reconstruction_error": compute_reconstruction_error(model_tensor, targets_tensor, normalize=True),
            "memorizer_reconstruction_error": compute_reconstruction_error(memorizer_tensor, targets_tensor, normalize=True)
        }

        if params["compute_final_tensor_ranks"]:
            tol = params["tensor_rank_tol"]
            model_tensor_rank = trm.find_cp_rank(model_tensor, tol=tol)
            memorizer_tensor_rank = trm.find_cp_rank(memorizer_tensor, tol=tol)
            targets_tensor_rank = trm.find_cp_rank(targets_tensor, tol=tol)

            metrics.update({"model_tensor_rank": model_tensor_rank,
                            "memorizer_tensor_rank": memorizer_tensor_rank,
                            "targets_tensor_rank": targets_tensor_rank})

        return metrics
