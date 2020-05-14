import itertools
from decimal import Decimal

import common.utils.logging as logging_utils
from trainers.tensor_completion_configurable_trainer import TensorCompletionConfigurableTrainer


def create_params_seq():
    base_params = {
        "experiment_name": "",
        "dataset_path": "data/tc/tensor_rank_1_fro_1_order_3_dim_8.pt",
        "num_samples": 300,
        "random_seed": -1,
        "output_dir": "outputs/tc/r1",
        "file_log": True,
        "store_checkpoints": True,
        "plot_metrics": True,
        "disable_gpu": False,
        "gpu_id": 0,
        "trainer_checkpoint": "",
        "epochs": 1000000,
        "validate_every": 100,
        "save_every_num_val": 5,
        "lr": 0.01,
        "use_adaptive_lr": True,
        "init_std": 0.001,
        "stop_on_zero_loss_tol": 1e-6,
        "tensor_rank_tol": 1e-6,
        "compute_final_tensor_ranks": True,
        "track_tensor_rank": False,
    }

    options = {
        "num_samples": [50, 100, 150, 200, 250, 300, 350, 400, 450, 511],
        "init_std": [0.01, 0.001]
    }

    repetitions = 5
    params_seq = []
    for _ in range(repetitions):
        params_seq.extend(__create_experiments_params_seq(base_params, options))

    for params in params_seq:
        dataset_desc = params["dataset_path"].split("/")[-1][:-3]  # Extracts dataset file name without extension
        init_std_str = f"{Decimal(params['init_std']):.0e}"

        experiment_name = f"c_{dataset_desc}_samples_{params['num_samples']}_init_{init_std_str}"
        params["experiment_name"] = experiment_name

    return params_seq


def __create_experiments_params_seq(base_params, options):
    param_names = options.keys()
    param_values = [options[param_name] for param_name in param_names]

    params_seq = []
    all_options_iterator = itertools.product(*param_values)
    for i, values in enumerate(all_options_iterator):
        params = base_params.copy()
        for param_name, param_value in zip(param_names, values):
            params[param_name] = param_value

        params_seq.append(params)

    return params_seq


def main():
    params_seq = create_params_seq()
    logging_utils.info(f"Starting run of {len(params_seq)} experiments.")

    for i, params in enumerate(params_seq):
        logging_utils.info(f"Starting fit for experiment {i + 1} of {len(params_seq)}")
        configurable_trainer = TensorCompletionConfigurableTrainer()
        configurable_trainer.fit(params)
        logging_utils.info(f"Finished fit for experiment {i + 1} of {len(params_seq)}")


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
