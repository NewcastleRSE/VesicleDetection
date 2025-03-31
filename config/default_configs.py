import yaml 
    
model_config = dict()
model_config["fmaps"] = 32
model_config["fmap_inc_factors"] = 5
model_config["downsample_factors"] = [tuple([1,1,1]), tuple([1,1,1])]
model_config["padding"] = 'valid'
model_config["constant_upsample"] = True 

training_config = dict()
training_config["clahe"] = True 
training_config["has_mask"] = False 
training_config["input_shape"] = (40,40,40)
training_config["augmentations"] = ["gp.SimpleAugment(transpose_only=(1,2))", 
                                    "gp.ElasticAugment((1,10,10), (0,0.1,0.1), (0,math.pi/2))"]
training_config["batch_size"] = 1 
training_config["checkpoint_path"] = None 
training_config["snapshot_every"] = 0 
training_config["iterations"] = 1000
training_config["val_every"] = 100
training_config["best_score_name"] = 'fscore_average' # Options: 'fscore_{1,2,average}', 'recall_{1,2,average}' or 'precision_{1,2,average}'
training_config["save_every"] = 100

post_processing_config = dict()
post_processing_config["combine_pos_neg"] = True
post_processing_config["maxima_threshold"] = 50 

tiff_to_zarr_train_config = dict()
tiff_to_zarr_train_config['path_to_raw_tiff_train'] = ''
tiff_to_zarr_train_config['path_to_raw_tiff_validate'] = ''
tiff_to_zarr_train_config['path_to_gt_PC_pos_tiff_train'] = ''
tiff_to_zarr_train_config['path_to_gt_PC_pos_tiff_validate'] = ''
tiff_to_zarr_train_config['path_to_gt_PC_neg_tiff_train'] = ''
tiff_to_zarr_train_config['path_to_gt_PC_neg_tiff_validate'] = ''
tiff_to_zarr_train_config['output_zarr_path'] = 'data/name/train'
tiff_to_zarr_train_config['attributes'] = {'resolution': (1,1,1), 'axes': ('z','y','x'), 'offset': (0,0,0)}

tiff_to_zarr_predict_config = dict()
tiff_to_zarr_predict_config['path_to_raw_tiff'] = ''
tiff_to_zarr_predict_config['output_zarr_path'] = 'data/name/predict'
tiff_to_zarr_predict_config['attributes'] = {'resolution': (1,1,1), 'axes': ('z','y','x'), 'offset': (0,0,0)}



def set_default_configs():

    with open("config/model_config.yaml", "w") as file_object:
        yaml.dump(model_config, file_object)
    
    with open("config/training_config.yaml", "w") as file_object:
        yaml.dump(training_config, file_object)

    with open("config/post_processing_config.yaml", "w") as file_object:
        yaml.dump(post_processing_config, file_object)
    
    with open("config/tiff_to_zarr_train_config.yaml", "w") as file_object:
        yaml.dump(tiff_to_zarr_train_config, file_object)
    
    with open("config/tiff_to_zarr_predict_config.yaml", "w") as file_object:
        yaml.dump(tiff_to_zarr_predict_config, file_object)


if __name__ == "__main__":
    set_default_configs()