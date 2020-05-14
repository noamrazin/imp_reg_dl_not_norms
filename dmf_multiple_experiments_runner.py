import common.utils.logging as logging_utils
from trainers.dln_matrix_sensing_configurable_trainer import DLNMatrixSensingConfigurableTrainer


def create_experiments_params_and_param_options_seq():
    params_seq = [
        {
            "experiment_name": "completion_2x2_c_1_1_0_exp",
            "dataset_path": "data/dmf/completion_2x2_w_1_1_0.pt",
            "num_samples": 3,
            "experiment_random_seed": -1,
            "output_dir": "outputs/dmf",
            "file_log": True,
            "store_checkpoints": True,
            "plot_metrics": True,
            "save_results": True,
            "disable_gpu": False,
            "gpu_id": 0,
            "depth": 3,
            "weight_init_type": "normal",
            "init_alpha": 8e-3,
            "use_balanced_init": False,
            "enforce_pos_det": True,
            "enforce_neg_det": False,
            "epochs": 5000000,
            "stop_on_zero_loss_tol": 1e-04,
            "validate_every": 50,
            "save_every_num_val": 100,
            "lr": 5e-3,
            "tracked_e2e_value_of": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "track_singular_values": True,
        }
    ]

    return params_seq


def main():
    params_seq = create_experiments_params_and_param_options_seq()
    logging_utils.info(f"Starting run of {len(params_seq)} experiments.")

    for i, params in enumerate(params_seq):
        logging_utils.info(f"Starting fit for experiment {i + 1} of {len(params_seq)}")
        configurable_trainer = DLNMatrixSensingConfigurableTrainer()
        configurable_trainer.fit(params)
        logging_utils.info(f"Finished fit for experiment {i + 1} of {len(params_seq)}")


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
