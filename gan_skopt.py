
import os
import json
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

import skopt
from skopt import gp_minimize, forest_minimize, callbacks, load
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver

from models.GAN_GPU import GAN

from utils.config import process_config
from utils.dirs import create_dirs, create_dir
from utils.args import get_args
from utils.logger import get_logger
from utils import factory

import tensorflow as tf
from bunch import Bunch

log = get_logger()

config, config_dict = process_config("configs/wgan_opt.json")

data = factory.create("data_loaders."+config.data['name'])(config)

config_opt = []
config_opt.append([Categorical(categories=[128, 256, 512, 1024],    name='d_num_filters'),          config.arch["d_num_filters"]])
config_opt.append([Categorical(categories=[False, True],            name='d_batch_norm'),           config.arch["d_batch_norm"]])
config_opt.append([Categorical(categories=[False, True],            name='g_batch_norm'),           config.arch["g_batch_norm"]])
config_opt.append([Categorical(categories=[3, 4, 5],                name='d_conv_kernal'),          config.arch["d_conv_kernal"]])
config_opt.append([Categorical(categories=[3, 4, 5],                name='g_conv_kernal'),          config.arch["g_conv_kernal"]])
config_opt.append([Categorical(categories=[3, 4],                   name='d_num_layers'),           config.arch["d_num_layers"]])
config_opt.append([Categorical(categories=[3, 4],                   name='g_num_layers'),           config.arch["g_num_layers"]])
config_opt.append([Real(low=5e-5, high=5e-4,                        name='optim_lr'),               config.model["optim_lr"]])
config_opt.append([Real(low=0, high=0.99,                           name='optim_beta1'),            config.model["optim_beta1"]])
config_opt.append([Real(low=0, high=0.99,                           name='optim_beta2'),            config.model["optim_beta2"]])
config_opt.append([Categorical(categories=[50, 70, 90, 100, 130],   name='input_noise_dim'),        config.model["input_noise_dim"]])
config_opt.append([Categorical(categories=[5, 7, 9, 10],            name='discriminator_iters'),    config.trainer["discriminator_iters"]])

dimensions = [x[0] for x in config_opt]

default_parameters = [x[1] for x in config_opt]

dims = [
    ["arch",    "d_num_filters",        0],
    ["arch",    "g_num_filters",        0],
    ["arch",    "d_batch_norm",         1],
    ["arch",    "g_batch_norm",         2], 
    ["arch",    "d_conv_kernal",        3],
    ["arch",    "g_conv_kernal",        4],
    ["arch",    "d_num_layers",         5],
    ["arch",    "g_num_layers",         6],
    ["model",   "optim_lr",             7], 
    ["model",   "optim_beta1",          8],
    ["model",   "optim_beta2",          9],
    ["model",   "input_noise_dim",      10],
    ["trainer", "discriminator_iters",  11],
]

max_tests = 20
N_test_samples = 10000

experiment_name =  config.exp_name + "_tune"
experiment_dir = os.path.join(os.getcwd(),'tuner_experiments/skopt/', experiment_name)

create_dir(experiment_dir)

def fitness(dimensions = dimensions):

    global experiment_dir
    global experiment_i

    global data
    global best_test
    global best_experiment_i
    global best_model_path
    global max_tests
    global N_test_samples

    global start_time
    global current_time
    global experiment_times

    current_dir = os.path.join(experiment_dir, "exp_%d"%experiment_i)
    create_dir(current_dir)
    
    log.info("####################################################################")
    log.info("experiment %d"%experiment_i)
    
    for c in dims:
        config[c[0]][c[1]] = dimensions[c[2]]
        log.info("{:30s}:{:20s}".format(c[1],str(dimensions[c[2]])))
    
    with open(os.path.join(current_dir, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    model = GAN(config)
    
    with tf.Session() as sess:
        
        sess.run(model.init)
        losses = None
        tests = None

        fake_labels = None
        if 'label' in config.data:
            fake_labels = data.sample_random_labels(N_test_samples)
        
        noise = [np.random.normal(0,1,[N_test_samples, config.model['input_noise_dim']]), fake_labels]
        
        for epoch in range(config.trainer['num_epochs']):
            log.info("epoch "+ str(epoch))
            losses_epoch = model.train_epoch(sess, data)
            
            fake_curves = model.generate_curves_gpu(sess, noise)
            
            fake_data = data.inverse_transform(fake_curves, fake_labels)

            mean_tests = data.test_similarity(fake_data)['tests']
            
            if losses is None:
                losses = losses_epoch
            else:
                for key in losses_epoch:
                    losses[key] = losses[key] + losses_epoch[key] 

            if tests is None:
                tests = mean_tests
            else:
                tests = pd.concat([tests, mean_tests])
            
            log.info("indicator distance : "+str(mean_tests.values.sum()))

            distance_to_data = mean_tests.values.sum()

            if distance_to_data < best_test:
                best_model_path = model.save(sess, checkpoint_dir=os.path.join(current_dir, "model"))
                best_test = distance_to_data
                best_experiment_i = experiment_i
                
            if mean_tests.values.sum()>max_tests and epoch >5:
                break

        losses = pd.DataFrame.from_dict(losses)
        losses.to_csv(os.path.join(current_dir, "log.csv"), sep=";")

        tests.to_csv(os.path.join(current_dir, "results.csv"), sep=";")
        
        log.info("distance: ")
        log.info(mean_tests)
        
    del model
    tf.reset_default_graph()

    experiment_time = datetime.now() - current_time
    experiment_times.append(experiment_time)
    log.info("experiment time : "+ str(experiment_time))    
    current_time = datetime.now()

    times = {}
    times["total_time"] = datetime.now() - start_time
    times["experiment_times"] = experiment_times
    times["best_model_path"] = best_model_path

    with open(os.path.join(experiment_dir, 'experiment_times.pkl'), 'bw') as fp:
        pickle.dump(times, fp)

    experiment_i+=1
    return distance_to_data

if(os.path.isfile(os.path.join(experiment_dir, 'experiment_times.pkl'))):
    with open(os.path.join(experiment_dir, 'experiment_times.pkl'), 'br') as fp:
        times = pickle.load(fp)
    start_time = datetime.now() - times['total_time']
    current_time = start_time
    experiment_times = times['experiment_times']
    best_model_path = times['best_model_path']
else :  
    start_time = datetime.now()
    current_time = datetime.now()
    experiment_times =[]
    best_model_path = ""
    

if(os.path.isfile(os.path.join(experiment_dir, "checkpoint.pkl"))):
    res = load(os.path.join(experiment_dir, "checkpoint.pkl"))
    x0 = res.x_iters
    y0 = res.func_vals
    experiment_i = len(y0)
    best_test = min(y0)
    best_experiment = y0.tolist().index(best_test)

else: 
    x0 = default_parameters
    y0 = None
    experiment_i = 0     
    best_experiment = 0
    best_test = 1

checkpoint_saver = CheckpointSaver(os.path.join(experiment_dir, "checkpoint.pkl"), compress=9)

search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=11,
                            callback=[checkpoint_saver],
                            x0= x0, #default_parameters
                            y0= y0)

times = {
    "total_time": datetime.now() - start_time,
    "experiment_times": experiment_times,
    "best_model_path": best_model_path
}

with open(os.path.join(experiment_dir, 'experiment_times.pkl'), 'bw') as fp:
    pickle.dump(times, fp)

skopt.dump(search_result, os.path.join(experiment_dir, "search_results"), store_objective=False)
    