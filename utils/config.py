import json
from bunch import Bunch
import os
import collections

base_config = os.path.join(os.getcwd(),"configs","base_config.json")

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    
    #with open(base_config, 'r') as config_file:
    #    base_dict = json.load(config_file)

    #config_dict = update(base_dict, config_dict)
    
    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    
    return config, config_dict

def process_config(json_file):
    config, config_dict = get_config_from_json(json_file)

    config.experiment_dir = os.path.join("experiments", config.exp_name)
    
    config.summary_dir = os.path.join("experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoint/")
    
    return config, config_dict

