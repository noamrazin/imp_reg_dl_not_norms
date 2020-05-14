import random
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from .configurable_trainer import ConfigurableTrainer
from .configurable_trainer import ConfigurableTrainerFitResult
from ..callbacks import Callback
from ..fit_output import FitOutput
from ..trainer import Trainer
from ...evaluation.evaluators import Evaluator, TrainEvaluator
from ...utils import module as module_utils


class ConfigurableTrainerBase(ConfigurableTrainer, ABC):
    """
    Base abstract class for implementing a ConfigurableTrainer. Currently automatically handles these parameters:
        - 'epochs': Validates it exists, and gives the param to the trainer.
        - 'random_seed': If exists, will use it to initialize torch, numpy and python random seeds.
        - 'gpu_id' and 'disable_gpu': Used to determine on which device to run training on, and then updates the state with the device.
        - 'trainer_checkpoint': If exists, will load the trainer checkpoint from the given path before starting training.
        - 'validate_every': Will run validation every this number of epochs.
    """

    def validate_params(self, params: dict):
        """
        Verifies the given params are valid, and contain the necessary parameters.
        :param params: dictionary of fit parameters.
        """
        if "epochs" not in params:
            raise ValueError("Missing 'epochs' parameter in the given fit parameters.")

    def initialize(self, params: dict, state: dict):
        """
        Runs any necessary initialization code. For example, initializes the random seed.
        :param params: dictionary of fit parameters.
        :param state: dictionary of the fit state.
        """
        if "random_seed" in params:
            random_seed = params["random_seed"]
            self.__set_initial_random_seed(random_seed)

        disable_gpu = params["disable_gpu"] if "disable_gpu" in params else False
        gpu_id = params["gpu_id"] if "gpu_id" in params else 0
        device = module_utils.get_device(disable_gpu, gpu_id)
        state["device"] = device

    def __set_initial_random_seed(self, random_seed: int):
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.random.manual_seed(random_seed)
            random.seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

    @abstractmethod
    def create_model(self, params: dict, state: dict) -> nn.Module:
        """
        Creates the model.
        :param params: dictionary of fit parameters.
        :param state: dictionary of the fit state.
        :return: The model.
        """
        raise NotImplementedError

    @abstractmethod
    def create_train_dataloader(self, params: dict, state: dict) -> torch.utils.data.DataLoader:
        """
        Creates the train dataloader.
        :param params: dictionary of fit parameters.
        :param state: dictionary of the fit state.
        :return: Train dataloader.
        """
        raise NotImplementedError

    @abstractmethod
    def create_train_and_validation_evaluators(self, model: nn.Module, device, params: dict, state: dict) -> Tuple[TrainEvaluator, Evaluator]:
        """
        Creates the train and validation evaluators.
        :param model: PyTorch model.
        :param device: device to use for evaluation.
        :param params: dictionary of fit parameters.
        :param state: dictionary of the fit state.
        :return: Train evaluator, Validation evaluator.
        """
        raise NotImplementedError

    @abstractmethod
    def create_trainer_callback(self, model: nn.Module, params: dict, state: dict) -> Callback:
        """
        Creates the callback for the trainer.
        :param model: PyTorch model.
        :param params: dictionary of fit parameters.
        :param state: dictionary of the fit state.
        :return: Callback to be called by the trainer during fitting.
        """
        raise NotImplementedError

    @abstractmethod
    def create_trainer(self, model: nn.Module, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, params: dict, state: dict) -> Trainer:
        """
        Creates a Trainer object for training the model.
        :param model: PyTorch model.
        :param train_evaluator: Train evaluator.
        :param val_evaluator: Validation evaluator.
        :param callback: Callback to be called during training.
        :param device: device to use for training.
        :param params: dictionary of fit parameters.
        :param state: dictionary of the fit state.
        :return: Trainer for training the model.
        """
        raise NotImplementedError

    @abstractmethod
    def create_fit_result(self, trainer: Trainer, fit_output: FitOutput, params: dict, state: dict) -> ConfigurableTrainerFitResult:
        """
        Creates the fit result from the fit output.
        :param trainer: Trainer object used to train the model.
        :param fit_output: FitOutput that is the result of fitting a Trainer object.
        :param params: dictionary of fit parameters.
        :param state: dictionary of the fit state.
        :return: ConfigurableTrainerFitResult fit result for the given fit output.
        """
        raise NotImplementedError

    def fit(self, params):
        self.validate_params(params)

        state = {}
        self.initialize(params, state)

        model = self.create_model(params, state)
        device = state["device"] if "device" in state else module_utils.get_device()
        model = model.to(device)

        train_dataloader = self.create_train_dataloader(params, state)
        train_evaluator, val_evaluator = self.create_train_and_validation_evaluators(model, device, params, state)

        callback = self.create_trainer_callback(model, params, state)
        trainer = self.create_trainer(model, train_evaluator, val_evaluator, callback, device, params, state)

        if "trainer_checkpoint" in params and params["trainer_checkpoint"]:
            trainer.load_state_dict(torch.load(params["trainer_checkpoint"], map_location=device))

        validate_every = params["validate_every"] if "validate_every" in params else 1
        fit_output = trainer.fit(train_dataloader, num_epochs=params["epochs"], validate_every=validate_every)
        return self.create_fit_result(trainer, fit_output, params, state)
