class BaseConfig:
    def __init__(self):
        # Paths
        self.cache_dir = "cache"
        self.data_folder = "data/BTP720"
        self.output_dir = "output/BTP720"

        # Data
        self.prepare_dataset = False
        self.k_folds = 10
        self.fold = 10

        # Model
        self.max_sequence_length = 1024
        self.pretrained_name = "QizhiPei/biot5-plus-large"
        self.accelerator = "cuda"
        self.trainable_layers = ["21", "22", "23"]

        # Hyperparameters
        self.seed = 0
        self.chosen_features = ["peptide", "selfies"]
        self.learning_rate = 0.0005
        self.per_device_train_batch_size = 8
        self.per_device_eval_batch_size = 128
        self.num_train_epochs = 50
        self.weight_decay = 0.01
        self.push_to_hub = False
        self.dropout = 0.05
        self.warmup_ratio = 0.05
        self.report_to = "none"
        self.metric_for_best_model = "eval_matthews_correlation"


config = BaseConfig()
