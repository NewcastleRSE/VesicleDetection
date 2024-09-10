import yaml 
import gunpowder as gp 
import math

class ModelConfigs:
    def __init__(self):
        with open("config/model_config.yaml", "r") as file_object:
            config = yaml.full_load(file_object)

        self.constant_upsample = config["constant_upsample"]
        self.downsample_factors = config["downsample_factors"]
        self.fmap_inc_factors = config["fmap_inc_factors"]
        self.fmaps = config["fmaps"]
        self.padding = config["padding"]
        
        for key, value in zip(config.keys(), config.values()):
             if not hasattr(ModelConfigs, key):
                setattr(ModelConfigs, key, value)

class TrainingConfigs:
    def __init__(self):
        with open("config/training_config.yaml", "r") as file_object:
            config = yaml.full_load(file_object)

        self.augmentations = [eval(aug) for aug in config['augmentations']]
        self.batch_size = config["batch_size"]
        self.checkpoint_path = config["checkpoint_path"]
        self.clahe = config['clahe']
        self.has_mask = config['has_mask']
        self.input_shape = config['input_shape']
        self.iterations = config["iterations"]
        self.snapshot_every = config['snapshot_every']
        
        for key, value in zip(config.keys(), config.values()):
             if not hasattr(TrainingConfigs, key):
                setattr(TrainingConfigs, key, value)

class PostProcessingConfigs:
    def __init__(self):
        with open("config/post_processing_config.yaml", "r") as file_object:
            config = yaml.full_load(file_object)

        self.combine_pos_neg = config["combine_pos_neg"]
        self.maxima_threshold = config["maxima_threshold"]
        
        for key, value in zip(config.keys(), config.values()):
             if not hasattr(PostProcessingConfigs, key):
                setattr(PostProcessingConfigs, key, value)


MODEL_CONFIG = ModelConfigs()
TRAINING_CONFIG = TrainingConfigs()
POST_PROCESSING_CONFIG = PostProcessingConfigs()
