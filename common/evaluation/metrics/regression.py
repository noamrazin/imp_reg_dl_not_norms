from .metric import AveragedMetric


class MSELoss(AveragedMetric):
    """
    MSE loss metric.
    """

    def __init__(self, reduction="mean", normalization_const=1):
        """
        :param reduction: reduction method param as supported by PyTorch MSELoss. Currently supports 'mean', 'sum' and 'none'
        :param normalization_const: Loss is divided by this constant.
        """
        super().__init__()
        self.reduction = reduction
        self.normalization_const = normalization_const

    def _calc_metric(self, y_pred, y):
        """
        Calculates the mean square error loss.
        :param y_pred: predictions.
        :param y: true values.
        :return: (Mean square error loss, num samples in input)
        """
        losses = (y_pred - y) ** 2
        loss = losses.mean() if self.reduction == "mean" else losses.sum()
        loss /= self.normalization_const
        return loss.item(), len(y)
