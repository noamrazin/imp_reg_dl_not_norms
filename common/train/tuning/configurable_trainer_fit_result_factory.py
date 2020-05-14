from typing import Tuple, Dict

import numpy as np

from ..fit_output import FitOutput
from ..tracked_value import TrackedValue
from ..tuning import ConfigurableTrainerFitResult


class ConfigurableTrainerFitResultFactory:
    """
    Static factory for creating ConfigurableTrainerFitResult objects from the returned FitResult of a Trainer.
    """

    @staticmethod
    def create_from_best_metric_score(metric_name: str, fit_output: FitOutput, largest=True) -> ConfigurableTrainerFitResult:
        """
        Creates a configurable model fit result with the best score value of the given metric from all of the training epochs.
        :param metric_name: name of the metric that defines the score.
        :param fit_output: result of the Trainer fit method.
        :param largest: Whether larger is better. If false then the score used will be the additive inverse.
        :return: ConfigurableTrainerFitResult with the score name, best score value, best score epoch and additional metadata with the rest of the
        metric values from the epoch that achieved the best score.
        """
        val_tracked_values = fit_output.val_tracked_values

        relevant_tracked_value = val_tracked_values[metric_name]
        has_metric_history = len(relevant_tracked_value.epoch_values) > 0
        score = np.max(relevant_tracked_value.epoch_values) if has_metric_history else relevant_tracked_value.current_value
        score = score if score is not None else -np.inf
        if not largest:
            score = -score

        argmax_score = np.argmax(relevant_tracked_value.epoch_values) if has_metric_history else -1
        best_score_epoch = relevant_tracked_value.epochs_with_values[argmax_score] if argmax_score != -1 else -1

        additional_metadata = ConfigurableTrainerFitResultFactory.__create_additional_metadata(fit_output, best_score_epoch)
        return ConfigurableTrainerFitResult(score, metric_name, best_score_epoch, additional_metadata)

    @staticmethod
    def create_from_last_metric_score(metric_name: str, fit_output: FitOutput, largest=True) -> ConfigurableTrainerFitResult:
        """
        Creates a configurable model fit result with the last score value of the given metric from the last training epoch.
        :param metric_name: name of the metric that defines the score.
        :param fit_output: result of the Trainer fit method.
        :param largest: Whether larger is better. If false then the score used will be the additive inverse.
        :return: ConfigurableModelFitResult with the score name, last score value, last score epoch or -1 if no epoch history is saved,
        and additional metadata with the rest of the metric values from the last epoch.
        """
        val_tracked_values = fit_output.val_tracked_values

        relevant_tracked_value = val_tracked_values[metric_name]
        score = relevant_tracked_value.current_value
        score = score if score is not None else -np.inf
        if not largest:
            score = -score

        score_epoch = relevant_tracked_value.epochs_with_values[-1] if relevant_tracked_value.epochs_with_values else -1

        additional_metadata = ConfigurableTrainerFitResultFactory.__create_additional_metadata(fit_output)
        return ConfigurableTrainerFitResult(score, metric_name, score_epoch, additional_metadata)

    @staticmethod
    def create_from_best_metric_with_prefix_score(metric_name_prefix: str, fit_output: FitOutput, largest=True) -> ConfigurableTrainerFitResult:
        """
        Creates a configurable model fit result with the best score value of the out of all the metrics that start with the given prefix
        from all of the training epochs.
        :param metric_name_prefix: prefix of the metric names that are considered when getting the best score.
        :param fit_output: result of the Trainer fit method.
        :param largest: Whether larger is better. If false then the score used will be the additive inverse.
        :return: ConfigurableModelFitResult with the score name, best score value, best score epoch and additional metadata with the rest of the
        metric values from the epoch that achieved the best score.
        """
        val_tracked_values = fit_output.val_tracked_values
        score, score_name, best_score_epoch = ConfigurableTrainerFitResultFactory.__get_best_metric_score_and_name_with_prefix(metric_name_prefix,
                                                                                                                               val_tracked_values)
        if not largest:
            score = -score

        additional_metadata = ConfigurableTrainerFitResultFactory.__create_additional_metadata(fit_output, best_score_epoch)
        return ConfigurableTrainerFitResult(score, score_name, best_score_epoch, additional_metadata)

    @staticmethod
    def create_from_last_metric_with_prefix_score(metric_name_prefix: str, fit_output: FitOutput, largest=True) -> ConfigurableTrainerFitResult:
        """
        Creates a configurable model fit result with the best last score value of the of the metrics that start with the given prefix.
        :param metric_name_prefix: prefix of the metric names that are considered when getting the best score.
        :param fit_output: result of the Trainer fit method.
        :param largest: Whether larger is better. If false then the score used will be the additive inverse.
        :return: ConfigurableModelFitResult with the score name, last score value, last score epoch or -1 if no epoch history is saved,
        and additional metadata with the rest of the metric values from the last epoch.
        """
        val_tracked_values = fit_output.val_tracked_values
        score, score_name, score_epoch = ConfigurableTrainerFitResultFactory.__get_best_last_metric_score_and_name_with_prefix(metric_name_prefix,
                                                                                                                               val_tracked_values)
        if not largest:
            score = -score

        additional_metadata = ConfigurableTrainerFitResultFactory.__create_additional_metadata(fit_output)
        return ConfigurableTrainerFitResult(score, score_name, score_epoch, additional_metadata)

    @staticmethod
    def __create_additional_metadata(fit_output: FitOutput, score_epoch: int = -1) -> dict:
        train_tracked_values = fit_output.train_tracked_values
        val_tracked_values = fit_output.val_tracked_values

        additional_metadata = {}
        if score_epoch != -1:
            additional_metadata.update(
                {f"Train {name}": tracked_value.epoch_values[tracked_value.epochs_with_values.index(score_epoch)]
                 for name, tracked_value in train_tracked_values.items()})
            additional_metadata.update(
                {f"Validation {name}": tracked_value.epoch_values[tracked_value.epochs_with_values.index(score_epoch)]
                 for name, tracked_value in val_tracked_values.items()})
        else:
            additional_metadata.update({f"Train {name}": tracked_value.current_value
                                        for name, tracked_value in train_tracked_values.items()})
            additional_metadata.update({f"Validation {name}": tracked_value.current_value
                                        for name, tracked_value in val_tracked_values.items()})

        if fit_output.exception_occured():
            additional_metadata["Exception"] = str(fit_output.exception)

        return additional_metadata

    @staticmethod
    def __get_best_metric_score_and_name_with_prefix(metric_name_prefix: str,
                                                     tracked_values: Dict[str, TrackedValue]) -> Tuple[float, str, int]:
        scores = []
        names = []
        best_score_epochs = []
        for name, tracked_value in tracked_values.items():
            if name.startswith(metric_name_prefix):
                has_metric_history = len(tracked_value.epoch_values) > 0
                score = np.max(tracked_value.epoch_values) if has_metric_history else tracked_value.current_value
                score = score if score is not None else -np.inf
                argmax_score = np.argmax(tracked_value.epoch_values) if has_metric_history else -1

                best_score_epochs.append(tracked_value.epochs_with_values[argmax_score] if has_metric_history else -1)
                scores.append(score)
                names.append(name)

        index_of_max = np.argmax(scores)
        return scores[index_of_max], names[index_of_max], best_score_epochs[index_of_max]

    @staticmethod
    def __get_best_last_metric_score_and_name_with_prefix(metric_name_prefix: str,
                                                          tracked_values: Dict[str, TrackedValue]) -> Tuple[float, str, int]:
        scores = []
        names = []
        score_epochs = []
        for name, tracked_value in tracked_values.items():
            if name.startswith(metric_name_prefix):
                score = tracked_value.current_value
                score = score if score is not None else -np.inf
                score_epochs.append(tracked_value.epochs_with_values[-1] if tracked_value.epochs_with_values else -1)
                scores.append(score)
                names.append(name)

        index_of_max = np.argmax(scores)
        return scores[index_of_max], names[index_of_max], score_epochs[index_of_max]
