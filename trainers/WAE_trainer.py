from base.base_trainer import BaseTrainer
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

from datetime import datetime

class WAETrainer(BaseTrainer):
    def __init__(self, sess, model, data, config,logger):
        super(WAETrainer, self).__init__(sess, model, data, config,logger)
        self.iters_per_epoch = int(data.train_size/self.config.trainer['batch_size'])


    def train(self):
        start_time = datetime.now()
        
        losses =None
        tests = None

        fake_labels = None
        if 'label' in self.config.data:
            fake_labels = self.data.sample_random_labels(10000)

        noise = [self.model.sample_pz(10000), fake_labels]
        
        min_value = 10

        if self.config.model['e_pretrain']:
            self.model.pretrain_encoder(self.sess, self.data)

        for epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.model.cur_epoch_tensor.eval(self.sess)+self.config.trainer['num_epochs'], 1):
            self.log.info("epoch : %d"%epoch)
            
            if self.config.model['lr_schedule'] == "manual":
                if epoch == 30:
                    self.model.decay = decay / 2.
                if epoch == 50:
                    self.model.decay = decay / 5.
                if epoch == 100:
                    self.model.decay = decay / 10.
            elif self.config.model['lr_schedule'] == "manual_smooth":
                enum = self.config.trainer['num_epochs']
                decay_t = np.exp(np.log(100.) / enum)
                self.model.decay = decay / decay_t

            elif self.config.model['lr_schedule'] != "plateau":
                assert type(self.config.model['lr_schedule']) == float
                decay = 1.0 * 10**(-epoch / float(self.config.model['lr_schedule']))

            loss = self.model.train_epoch(self.sess, self.data)
            self.sess.run(self.model.increment_cur_epoch_tensor)

            if losses is None:
                losses = loss
            else:
                for key in loss:
                    losses[key] = losses[key] + loss[key] 

            reconstruct_results = self.model.reconstruct_curves(self.sess, data.test_data)
            fake_curves = self.model.generate_curves(self.sess, noise)

            if 'label' in self.config.data:
                fake_data = self.data.inverse_transform(fake_curves, fake_labels)
            else:
                fake_data = self.data.inverse_transform(fake_curves)
            mean_tests = self.data.test_similarity(fake_data)['tests']
            
            self.log.info("indicator distance : "+str(mean_tests.values.sum()))
            
            mean_tests['test_loss'] = reconstruct_results['rec_loss']
            
            if(mean_tests.values.sum() < min_value):
                min_value = mean_tests.values.sum()
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
            