from ..trainer import Trainer
from ...evaluation.evaluators.evaluator import VoidEvaluator
from ...utils import module as module_utils


class SupervisedTrainer(Trainer):
    """
    Trainer for regular supervised task of predicting y given x (classification or regression).
    """

    def __init__(self, model, optimizer, loss_fn, train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(), callback=None,
                 device=module_utils.get_device()):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn

    def batch_update(self, batch_num, batch):
        self.optimizer.zero_grad()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)

        loss = self.loss_fn(y_pred, y)
        loss.backward()

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "y_pred": y_pred.detach(),
            "y": y
        }
