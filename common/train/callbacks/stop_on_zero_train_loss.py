from typing import Callable

from common.train.callbacks import Callback
from common.train.stop_fit_iteration import StopFitIteration


class StopOnZeroTrainLoss(Callback):
    """
    Stops training if loss converges to 0.
    """

    def __init__(self, train_loss_fn: Callable, cooldown: int = 0, patience: int = 10, tol: float = 1e-6):
        """
        :param train_loss_fn: Function that given a Trainer object returns the last training loss during training.
        :param cooldown: Number of epochs from start of training before checking for convergence to 0.
        :param patience: Number of epochs loss has to remain 0 before stopping.
        :param tol: Tolerance value that below it the loss will be considered 0.
        """
        self.train_loss_fn = train_loss_fn
        self.cooldown = cooldown
        self.patience = patience
        self.tol = tol

        self.num_zero_train_loss_in_a_row = 0

    def on_epoch_validation_end(self, trainer, metric_values):
        if trainer.epoch < self.cooldown:
            return

        self.__check_convergence_to_zero(trainer)

    def __check_convergence_to_zero(self, trainer):
        loss = self.train_loss_fn(trainer)
        if loss < self.tol:
            self.num_zero_train_loss_in_a_row += 1
        else:
            self.num_zero_train_loss_in_a_row = 0

        if self.num_zero_train_loss_in_a_row > self.patience:
            self.__stop_fitting(trainer.epoch)

    def __stop_fitting(self, epoch):
        raise StopFitIteration(f"Stopping at end of epoch {epoch} because the train loss was below {self.tol} for "
                               f"{self.num_zero_train_loss_in_a_row} validation epochs in a row")
