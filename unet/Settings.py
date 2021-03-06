
class Settings(object):

    def __init__(self):

        # Dataset arguments ##############################################
        self.dataset_args = dict()

        self.dataset_args["data_path_column"] = "image"
        self.dataset_args["mask_path_column"] = "mask"

        self.dataset_args["resize_shape"] = (256, 256)

        self.dataset_args["postprocessing_thr"] = 0.5
        self.dataset_args["predictions_dir"] = "predictions"

        # DataSplitter arguments ##############################################
        self.data_splitter_args = dict()

        self.data_splitter_args["data_definition_file_path"] = "/data/eytank/datasets/hubmap_kidney/ds_tile1024_step1024_sampled.csv"

        self.data_splitter_args["folds_num"] = 6
        self.data_splitter_args["data_random_seed"] = 1509

        self.data_splitter_args["train_data_file_name"] = "train_data.json"
        self.data_splitter_args["val_data_file_name"] = "val_data.json"
        self.data_splitter_args["test_data_file_name"] = "test_data.json"

        self.data_splitter_args["split_to_groups"] = False
        self.data_splitter_args["group_column"] = ""
        self.data_splitter_args["group_ids"] = None

        self.data_splitter_args["leave_out"] = True  # allows to choose for test data with unique values of 'self.leave_out_column'
        self.data_splitter_args["leave_out_column"] = "raw_image_id"
        self.data_splitter_args["leave_out_values"] = None

        # Sequence arguments
        self.sequence_args = dict()

        self.sequence_args["batch_size"] = 16
        self.sequence_args["apply_augmentations"] = True

        self.sequence_args["multi_input"] = False
        self.sequence_args["multi_output"] = False
        self.sequence_args["inputs_num"] = 1
        self.sequence_args["outputs_num"] = 1

        self.sequence_args["subsample"] = False
        self.sequence_args["subsampling_param"] = ""
        self.sequence_args["oversample"] = False
        self.sequence_args["oversampling_param"] = ""

        # Model arguments ##############################################
        self.model_args = dict()

        # Base model arguments
        self.model_args["model_name"] = "unet_3rd_party"
        self.model_args["epochs"] = 3000
        self.model_args["prediction_batch_size"] = 64

        # UNet arguments
        self.model_args["backbone"] = "efficientnetb4"
        self.model_args["input_shape"] = (256, 256, 3)
        self.model_args["num_classes"] = 1
        self.model_args["output_activation"] = "sigmoid"
        self.model_args["encoder_weights"] = "imagenet"

        # Regularizer arguments
        self.model_args["regularizer_args"] = dict()
        self.model_args["regularizer_args"]["regularizer_name"] = "l1_l2_regularizer"
        self.model_args["regularizer_args"]["l1_reg_factor"] = 0
        self.model_args["regularizer_args"]["l2_reg_factor"] = 0

        # Losses arguments
        self.model_args["loss_args"] = list()

        self.model_args["loss_args"].append(dict())
        self.model_args["loss_args"][0]["metric_name"] = "segmentation_metric"
        self.model_args["loss_args"][0]["dice"] = True
        self.model_args["loss_args"][0]["loss"] = True
        self.model_args["loss_args"][0]["loss_weight"] = 1

        # Optimizer arguments
        self.model_args["optimizer_args"] = dict()
        self.model_args["optimizer_args"]["optimizer_name"] = "adam_optimizer"
        self.model_args["optimizer_args"]["learning_rate"] = 1e-3

        # Metrics arguments
        self.model_args["metrics_args"] = list()

        # Metrics for first output
        self.model_args["metrics_args"].append(list())

        self.model_args["metrics_args"][0].append(dict())
        self.model_args["metrics_args"][0][0]["metric_name"] = "segmentation_metric"
        self.model_args["metrics_args"][0][0]["dice"] = True

        self.model_args["metrics_args"][0].append(dict())
        self.model_args["metrics_args"][0][1]["metric_name"] = "segmentation_metric"
        self.model_args["metrics_args"][0][1]["precision"] = True

        self.model_args["metrics_args"][0].append(dict())
        self.model_args["metrics_args"][0][2]["metric_name"] = "segmentation_metric"
        self.model_args["metrics_args"][0][2]["recall"] = True

        # Callback arguments
        self.model_args["callbacks_args"] = list()

        # Checkpoint callback arguments
        self.model_args["callbacks_args"].append(dict())
        self.model_args["callbacks_args"][0]["callback_name"] = "checkpoint_callback"
        self.model_args["callbacks_args"][0]["checkpoint_path"] = ""  # will be set by training script
        self.model_args["callbacks_args"][0]["save_best_only"] = True
        self.model_args["callbacks_args"][0]["monitor"] = "val_loss"

        # CSV Logger callback arguments
        self.model_args["callbacks_args"].append(dict())
        self.model_args["callbacks_args"][1]["callback_name"] = "csv_logger_callback"
        self.model_args["callbacks_args"][1]["training_log_path"] = ""  # will be set by training script

        # Early stopping callback arguments
        self.model_args["callbacks_args"].append(dict())
        self.model_args["callbacks_args"][2]["callback_name"] = "early_stopping_callback"
        self.model_args["callbacks_args"][2]["patience"] = 40
        self.model_args["callbacks_args"][2]["monitor"] = "val_loss"

        # Reduce learning rate callback arguments
        self.model_args["callbacks_args"].append(dict())
        self.model_args["callbacks_args"][3]["callback_name"] = "reduce_lr_on_plateau"
        self.model_args["callbacks_args"][3]["reduce_factor"] = 0.8
        self.model_args["callbacks_args"][3]["patience"] = 3
        self.model_args["callbacks_args"][3]["min_lr"] = 1e-6
        self.model_args["callbacks_args"][3]["monitor"] = "val_loss"

        # Logger arguments ##############################################
        self.logger_args = dict()

        self.logger_args["message_format"] = "%(message)s"
        self.logger_args["file_name"] = "results.log"

        # Output settings
        self.simulation_folder = "/data/eytank/simulations/hubmap_kidney/2021.03.27_smefficientnetb4_updateddata"
        self.training_log_name = "metrics.log"
        self.settings_file_name = "../Settings.py"
        self.saved_model_name = "model"

        # ClearML settings
        self.clear_ml = False
        self.clear_ml_project_name = ""
        self.clear_ml_task_name = ""
        self.clear_ml_connect_frameworks = {}

        # Test settings
        self.test_simulation = True
        self.get_original_data = True
        self.save_test_data = True

        # Inference settings
        self.inference_data_pattern = ""

        # Model training settings
        self.training_folds = [0, 1]
        self.load_model = False
        self.load_model_path = ""  # list for train/test, string for inference

        self.logs_dir = self.simulation_folder
        self.log_message = "Kidney glomeruli segmentation\n" \
                           "simple_converge 0.5.7\n" \
                           "'Segmentation Models' UNet architecture with EfficientNetB4 pretrained backbone\n" \
                           "Using updated dataset\n" \
                           "Using overlapped images during training\n" \
                           "Using basic augmentations: flip left-right, flip up-down, rotations to 90/180/270 degrees"

        self.plot_metrics = ["loss", "dice_metric", "recall_metric", "precision_metric"]
