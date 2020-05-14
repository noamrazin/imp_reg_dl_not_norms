import common.utils.module as module_utils
from common.evaluation.evaluators import VoidEvaluator
from common.train.trainer import Trainer
from models.deep_linear_net import DeepLinearNet


class DLNMatrixSensingTrainer(Trainer):

    def __init__(self, model: DeepLinearNet, optimizer, loss_fn, train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(), callback=None,
                 device=module_utils.get_device()):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn

    def batch_update(self, batch_num, batch):
        self.optimizer.zero_grad()

        A, y = batch
        A = A.to(self.device)
        y = y.to(self.device)

        end_to_end_matrix = self.model.compute_prod_matrix()
        y_pred = (A * end_to_end_matrix.unsqueeze(0)).sum(dim=(1, 2))

        loss = self.loss_fn(y_pred, y)
        loss.backward()

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "y_pred": y_pred.detach(),
            "y": y
        }
