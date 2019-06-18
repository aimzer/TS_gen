from base.base_trainer import BaseTrainer
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

from datetime import datetime

class GANTrainer(BaseTrainer):
    def __init__(self, sess, model, data, config,logger):
        super(GANTrainer, self).__init__(sess, model, data, config,logger)
        self.iters_per_epoch = int(data.train_size/self.config.trainer['discriminator_iters']/self.config.trainer['batch_size'])


    def train(self):
        losses_D_real = [] 
        losses_D_fake = []
        losses_D = []
        losses_G = []
        losses_M = []
        losses = None
        start_time = datetime.now()

        tests = None

        fake_labels = None
        if 'label' in self.config.data:
            fake_labels = self.data.sample_random_labels(10000)


        noise = [self.model.sample_noise(10000), fake_labels]
        
        min_value = 10

        for epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.model.cur_epoch_tensor.eval(self.sess)+self.config.trainer['num_epochs'], 1):
            self.log.info("epoch : %d"%epoch)
            loss = self.model.train_epoch(self.sess, self.data)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            if losses is None:
                losses = loss
            else:
                for key in loss:
                    losses[key] = losses[key] + loss[key] 

            fake_curves = self.model.generate_curves_gpu(self.sess, noise)

            if 'label' in self.config.data:
                fake_data = self.data.inverse_transform(fake_curves, fake_labels)
            else:
                fake_data = self.data.inverse_transform(fake_curves)
            mean_tests = self.data.test_similarity(fake_data)['tests']
            
            mean_res = mean_tests.values.sum()/self.config.model['input_label_dim'] if 'label' in self.config.data else mean_tests.values.sum() 
            self.log.info("indicator distance : "+str(mean_res))
            
            if(mean_res < min_value):
                min_value = mean_res
                self.model.save(self.sess)
            
            if tests is None:
                tests = mean_tests
            else:
                tests = pd.concat([tests, mean_tests])
        

        self.model.save(self.sess)

        losses = pd.DataFrame.from_dict(losses)
        
        if os.path.isfile(os.path.join(self.config.summary_dir, "log.csv")):
            prev_losses = pd.read_csv(os.path.join(self.config.summary_dir, "log.csv"), sep=";")
            losses = pd.concat([prev_losses, losses], axis=0)
        
        losses.to_csv(os.path.join(self.config.summary_dir, "log.csv"), sep=";")
        tests.to_csv(os.path.join(self.config.summary_dir, "results.csv"), sep=";")
        delta_time = datetime.now() - start_time

        self.log.info("experiment time : "+ str(delta_time))
            

    def train_epoch(self):
        pass

    def train_step(self):
        pass