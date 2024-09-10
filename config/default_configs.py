import yaml 
import gunpowder as gp 
import math
import ast 
    
model_config = dict()
model_config["fmaps"] = 32
model_config["fmap_inc_factors"] = 5
model_config["downsample_factors"] = [tuple([1,2,2]), tuple([1,2,2])]
model_config["padding"] = 'valid'
model_config["constant_upsample"] = True 

training_config = dict()
training_config["clahe"] = True 
training_config["has_mask"] = True 
training_config["input_shape"] = (30,96,96)
training_config["augmentations"] = ["gp.SimpleAugment(transpose_only=(1,2))", 
                                    "gp.ElasticAugment((1,10,10), (0,0.1,0.1), (0,math.pi/2))"]
training_config["batch_size"] = 1 
training_config["checkpoint_path"] = None 
training_config["snapshot_every"] = 0 
training_config["iterations"] = 100

post_processing_config = dict()
post_processing_config["combine_pos_neg"] = True
post_processing_config["maxima_threshold"] = 50 

def set_default_configs():

    with open("config/model_config.yaml", "w") as file_object:
        yaml.dump(model_config, file_object)
    
    with open("config/training_config.yaml", "w") as file_object:
        yaml.dump(training_config, file_object)

    with open("config/post_processing_config.yaml", "w") as file_object:
        yaml.dump(post_processing_config, file_object)


if __name__ == "__main__":
    set_default_configs()