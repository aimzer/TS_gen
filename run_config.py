
import os
import json

import tensorflow as tf

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.args import get_args
from utils import factory


def main():
    # capture the config path from the run arguments
    # then process the json configuration fill

    try:
        args = get_args()
        config, config_dict = process_config(args.config)
        
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    with open(os.path.join(config.experiment_dir, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    print('Create the data generator.')
    data = factory.create("data_loaders."+config.data['name'])(config)
    
    print('Create the model.')
    model = factory.create("models."+config.model['name'])(config)

    # create tensorflow session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    # create tensorboard logger
    logger = Logger(sess, config)
    
    print('Create the trainer')
    trainer = factory.create("trainers."+config.trainer['name'])(sess, model, data, config, logger)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()