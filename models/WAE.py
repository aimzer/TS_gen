
import time
import numpy as np
import tensorflow as tf

from archs.wae_nets import encoder, decoder, z_adversary
from base.base_model import BaseModel

import datetime

class WAE(BaseModel):

    def __init__(self, config):
        super(WAE, self).__init__(config)

        self.config = config

        self.add_inputs_placeholders()

        self.add_training_placeholders()

        self.build_model()

        self.add_least_gaussian2d_ops()

        self.add_optimizers()

        self.add_savers()

        self.init = tf.global_variables_initializer()

    def add_inputs_placeholders(self):
        config = self.config

        data = tf.placeholder(
            tf.float32, [None, config.model['input_curve_dim']], name='real_points_ph')
        noise = tf.placeholder(
            tf.float32, [None, config.model['zdim']], name='noise_ph')

        self.sample_points = data
        self.sample_noise = noise

    def add_training_placeholders(self):
        config = self.config

        self.lr_decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        self.wae_lambda = tf.placeholder(tf.float32, name='lambda_ph')
        self.is_training = tf.placeholder(tf.bool, name='is_training_ph')

    def pretrain_loss(self):
        config = self.config
        # Adding ops to pretrain the encoder so that mean and covariance
        # of Qz will try to match those of Pz
        mean_pz = tf.reduce_mean(self.sample_noise, axis=0, keepdims=True)
        mean_qz = tf.reduce_mean(self.encoded, axis=0, keepdims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
        cov_pz = tf.matmul(self.sample_noise - mean_pz,
                           self.sample_noise - mean_pz, transpose_a=True)
        cov_pz /= config.model['e_pretrain_sample_size'] - 1.
        cov_qz = tf.matmul(self.encoded - mean_qz,
                           self.encoded - mean_qz, transpose_a=True)
        cov_qz /= config.model['e_pretrain_sample_size'] - 1.
        cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
        return mean_loss + cov_loss

    def pretrain_encoder(self, sess, data_gen):
        config = self.config
        steps_max = 200
        batch_size = config.model['e_pretrain_sample_size']
        for step in range(steps_max):
            batch_noise =  self.sample_pz(batch_size)
            batch = next(data_gen.next_train_batch(self.config.trainer['batch_size']))
            
            [_, loss_pretrain] = sess.run(
                [self.pretrain_opt,
                 self.loss_pretrain],
                feed_dict={self.sample_points: batch['data'],
                           self.sample_noise: batch_noise,
                           self.is_training: True})

            if config.trainer['verbose']:
                self.log.info('Step %d/%d, loss=%f' % (
                    step, steps_max, loss_pretrain))

            if loss_pretrain < 0.1:
                break


    def build_model(self):

        config = self.config

        sample_size = tf.shape(self.sample_points)[0]

        # Encode the content of sample_points placeholder
        with tf.variable_scope("encoder", reuse=False):
            res = encoder(config, inputs=self.sample_points,
                        is_training=self.is_training)
        
        if config.model['e_noise'] in ('deterministic', 'implicit', 'add_noise'):
            self.enc_mean, self.enc_sigmas = None, None
            if config.model['e_noise'] == 'implicit':
                self.encoded, self.encoder_A = res
            else:
                self.encoded, _ = res

        elif config.model['e_noise'] == 'gaussian':
            # Encoder outputs means and variances of Gaussian
            enc_mean, enc_sigmas = res[0]
            enc_sigmas = tf.clip_by_value(enc_sigmas, -50, 50)
            self.enc_mean, self.enc_sigmas = enc_mean, enc_sigmas
            if config.trainer['verbose']:
                self.add_sigmas_debug()

            eps = tf.random_normal((sample_size, config.model['zdim']),
                                   0., 1., dtype=tf.float32)
            self.encoded = self.enc_mean + tf.multiply(
                eps, tf.sqrt(1e-8 + tf.exp(self.enc_sigmas)))
            # self.encoded = self.enc_mean + tf.multiply(
            #     eps, tf.exp(self.enc_sigmas / 2.))

        # Decode the points encoded above (i.e. reconstruct)
        with tf.variable_scope("decoder", reuse=False):
            self.reconstructed, self.reconstructed_logits = \
                    decoder(config, noise=self.encoded,
                            is_training=self.is_training)

        # Decode the content of sample_noise
        with tf.variable_scope("decoder", reuse=True):
            self.decoded, self.decoded_logits = \
                decoder(config, noise=self.sample_noise,
                        is_training=self.is_training)
        

        # -- Objectives, losses, penalties

        self.penalty, self.loss_gan = self.matching_penalty()
        self.loss_reconstruct = self.reconstruction_loss(
            self.config, self.sample_points, self.reconstructed)
        self.wae_objective = self.loss_reconstruct + self.wae_lambda * self.penalty

        # Extra costs if any
        #if 'w_aef' in config and config.model['w_aef'] > 0:
        #    improved_wae.add_aefixedpoint_cost(config, self)

        if config.model['e_pretrain']:
            self.loss_pretrain = self.pretrain_loss()
        else:
            self.loss_pretrain = None

    def add_savers(self):
        config = self.config
        saver = tf.train.Saver(max_to_keep = config.model['max_to_keep'])
        tf.add_to_collection('real_points_ph', self.sample_points)
        tf.add_to_collection('noise_ph', self.sample_noise)
        tf.add_to_collection('is_training_ph', self.is_training)
        if self.enc_mean is not None:
            tf.add_to_collection('encoder_mean', self.enc_mean)
            tf.add_to_collection('encoder_var', self.enc_sigmas)
        if config.model['e_noise'] == 'implicit':
            tf.add_to_collection('encoder_A', self.encoder_A)
        tf.add_to_collection('encoded', self.encoded)
        tf.add_to_collection('decoded', self.decoded)
        tf.add_to_collection('reconstruct', self.reconstructed)
        tf.add_to_collection('match_loss', self.penalty)
        tf.add_to_collection('reconstruct_loss', self.loss_reconstruct)
    
        if self.loss_gan:
            tf.add_to_collection('disc_logits_Pz', self.loss_gan[1])
            tf.add_to_collection('disc_logits_Qz', self.loss_gan[2])

        tf.add_to_collection('ae_training_op', self.ae_opt)

        if config.model['z_test'] == 'gan':
            tf.add_to_collection('gan_training_op', self.z_adv_opt)

        if config.model['e_pretrain']:
            tf.add_to_collection('ae_pretraining_op', self.pretrain_opt)
            
        
        self.saver = saver

    def add_least_gaussian2d_ops(self):
        """ Add ops searching for the 2d plane in z_dim hidden space
            corresponding to the 'least Gaussian' look of the sample
        """

        config = self.config

        with tf.variable_scope('leastGaussian2d'):
            # Projection matrix which we are going to tune
            sample = tf.placeholder(
                tf.float32, [None, config.model['zdim']], name='sample_ph')
            v = tf.get_variable(
                "proj_v", [config.model['zdim'], 1],
                tf.float32, tf.random_normal_initializer(stddev=1.))
            u = tf.get_variable(
                "proj_u", [config.model['zdim'], 1],
                tf.float32, tf.random_normal_initializer(stddev=1.))
            npoints = tf.cast(tf.shape(sample)[0], tf.int32)

            # First we need to make sure projection matrix is orthogonal

            v_norm = tf.nn.l2_normalize(v, 0)
            dotprod = tf.reduce_sum(tf.multiply(u, v_norm))
            u_ort = u - dotprod * v_norm
            u_norm = tf.nn.l2_normalize(u_ort, 0)
            Mproj = tf.concat([v_norm, u_norm], 1)
            sample_proj = tf.matmul(sample, Mproj)
            a = tf.eye(npoints)
            a -= tf.ones([npoints, npoints]) / tf.cast(npoints, tf.float32)
            b = tf.matmul(sample_proj, tf.matmul(a, a), transpose_a=True)
            b = tf.matmul(b, sample_proj)
            # Sample covariance matrix
            covhat = b / (tf.cast(npoints, tf.float32) - 1)
            gcov = config.model['pz_scale'] ** 2.  * tf.eye(2)
            # l2 distance between sample cov and the Gaussian cov
            projloss =  tf.reduce_sum(tf.square(covhat - gcov))
            # Also account for the first moment, i.e. expected value
            projloss += tf.reduce_sum(tf.square(tf.reduce_mean(sample_proj, 0)))
            # We are maximizing
            projloss = -projloss
            optim = tf.train.AdamOptimizer(0.001, 0.9)
            optim = optim.minimize(projloss, var_list=[v, u])

        self.proj_u = u_norm
        self.proj_v = v_norm
        self.proj_sample = sample
        self.proj_covhat = covhat
        self.proj_loss = projloss
        self.proj_opt = optim

    def matching_penalty(self):
        config = self.config
        loss_gan = None
        sample_qz = self.encoded
        sample_pz = self.sample_noise
        if config.model['z_test'] == 'gan':
            loss_gan, loss_match = self.gan_penalty(sample_qz, sample_pz)
        elif config.model['z_test'] == 'mmd':
            loss_match = self.mmd_penalty(sample_qz, sample_pz)
        elif config.model['z_test'] == 'mmdpp':
            loss_match = improved_wae.mmdpp_penalty(
                config, self, sample_pz)
        elif config.model['z_test'] == 'mmdppp':
            loss_match = improved_wae.mmdpp_1d_penalty(
                config, self, sample_pz)
        else:
            assert False, 'Unknown penalty %s' % config.model['z_test']
        return loss_match, loss_gan

    def mmd_penalty(self, sample_qz, sample_pz):
        config = self.config
        sigma2_p = config.model['pz_scale'] ** 2
        kernel = config.model['mmd_kernel']
        n = tf.shape(sample_qz)[0] 
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods
        

        if kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

            if config.trainer['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            res1 = tf.exp( - distances_qz / 2. / sigma2_k)
            res1 += tf.exp( - distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp( - distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':

            if config.model['pz'] == 'normal':
                Cbase = 2. * config.model['zdim'] * sigma2_p
            elif config.model['pz'] == 'sphere':
                Cbase = 2.
            elif config.model['pz'] == 'uniform':
                Cbase = config.model['zdim']
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
                
        return stat

    def gan_penalty(self, sample_qz, sample_pz):
        config = self.config
        # Pz = Qz test based on GAN in the Z space
        with tf.variable_scope('z_adversary', reuse=False):
            logits_Pz = z_adversary(config, sample_pz)
        with tf.variable_scope('z_adversary', reuse=True):
            logits_Qz = z_adversary(config, sample_qz)
        
        loss_Pz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Pz, labels=tf.ones_like(logits_Pz)))
        loss_Qz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Qz, labels=tf.zeros_like(logits_Qz)))
        loss_Qz_trick = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Qz, labels=tf.ones_like(logits_Qz)))
        loss_adversary = self.wae_lambda * (loss_Pz + loss_Qz)
        # Non-saturating loss trick
        loss_match = loss_Qz_trick
        return (loss_adversary, logits_Pz, logits_Qz), loss_match

    @staticmethod
    def reconstruction_loss(config, real, reconstr):
        # real = self.sample_points
        # reconstr = self.reconstructed
        axi = [1]
        if config.model['cost'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=axi)
            loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        elif config.model['cost'] == 'l2_with_deriv':
            # c(x,y) = ||x - y||_2
            d_recon = reconstr[:,1:-2,:] - 0.5*reconstr[:,0:-3,:] - 0.5*reconstr[:,2:-1,:]
            d_real = real[:,1:-2,:] - 0.5*real[:,0:-3,:] - 0.5*real[:,2:-1,:]
            d2_recon = reconstr[:,2:-3,:] - 0.5*reconstr[:,0:-5,:] - 0.5*reconstr[:,4:-1,:]
            d2_real = real[:,2:-3,:] - 0.5*real[:,0:-5,:] - 0.5*real[:,4:-1,:]
            d2_loss = tf.reduce_sum(tf.square(d2_real - d2_recon), axis=axi)
            d_loss = tf.reduce_sum(tf.square(d_real - d_recon), axis=axi)
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=axi)
            loss = 0.2 * (tf.reduce_mean(tf.sqrt(1e-08 + loss)) + 
                          config.model['deriv_lambda'] * tf.reduce_mean(tf.sqrt(1e-08 + d_loss)) +
                          config.model['deriv_lambda'] * tf.reduce_mean(tf.sqrt(1e-08 + d2_loss)))
        elif config.model['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=axi)
            loss = 0.05 * tf.reduce_mean(loss)
        elif config.model['cost'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss = tf.reduce_sum(tf.abs(real - reconstr), axis=axi)
            loss = 0.02 * tf.reduce_mean(loss)
        else:
            assert False, 'Unknown cost function %s' % config.model['cost']
        return loss

    def optimizer(self, lr, decay=1.):
        config = self.config
        lr *= decay
        if config.model["optim"] == "sgd":
            return tf.train.GradientDescentOptimizer(lr)
        elif config.model["optim"] == "adam":
            return tf.train.AdamOptimizer(lr, beta1=config.model["optim_beta1"])
        else:
            assert False, 'Unknown optimizer.'

    def add_optimizers(self):
        config = self.config
        lr = config.model['optim_lr']
        lr_adv = config.model['optim_lr_adv']
        z_adv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='z_adversary')
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        ae_vars = encoder_vars + decoder_vars

        if config.trainer['verbose']:
            self.log.info('Param num in AE: %d' % \
                    np.sum([np.prod([int(d) for d in v.get_shape()]) \
                    for v in ae_vars]))

        # Auto-encoder optimizer
        opt = self.optimizer(lr, self.lr_decay)
        self.ae_opt = opt.minimize(loss=self.wae_objective,
                              var_list=encoder_vars + decoder_vars, global_step=self.global_step)

        # Discriminator optimizer for WAE-GAN
        if config.model['z_test'] == 'gan':
            opt = self.optimizer(lr_adv, self.lr_decay)
            self.z_adv_opt = opt.minimize(
                loss=self.loss_gan[0], var_list=z_adv_vars)
        else:
            self.z_adv_opt = None

        # Encoder optimizer
        if config.model['e_pretrain']:
            opt = self.optimizer(lr)
            self.pretrain_opt = opt.minimize(loss=self.loss_pretrain,
                                             var_list=encoder_vars)
        else:
            self.pretrain_opt = None


    
    def sample_pz(self, num=100):
        config = self.config
        noise = None
        distr = config.model['pz']
        if distr == 'uniform':
            noise = np.random.uniform(
                -1, 1, [num, config.model["zdim"]]).astype(np.float32)
        elif distr in ('normal', 'sphere'):
            mean = np.zeros(config.model["zdim"])
            cov = np.identity(config.model["zdim"])
            noise = np.random.multivariate_normal(
                mean, cov, num).astype(np.float32)
            if distr == 'sphere':
                noise = noise / np.sqrt(
                    np.sum(noise * noise, axis=1))[:, np.newaxis]
        return config.model['pz_scale'] * noise

    def least_gaussian_2d(self, X):
        """
        Given a sample X of shape (n_points, n_z) find 2d plain
        such that projection looks least Gaussian
        """
        config = self.config
        with self.sess.as_default(), self.sess.graph.as_default():
            sample = self.proj_sample
            optim = self.proj_opt
            loss = self.proj_loss
            u = self.proj_u
            v = self.proj_v

            covhat = self.proj_covhat
            proj_mat = tf.concat([v, u], 1).eval()
            dot_prod = -1
            best_of_runs = 10e5 # Any positive value would do
            updated = False
            for _ in xrange(3):
                # We will run 3 times from random inits
                loss_prev = 10e5 # Any positive value would do
                proj_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='leastGaussian2d')
                self.sess.run(tf.variables_initializer(proj_vars))
                step = 0
                for _ in xrange(5000):
                    self.sess.run(optim, feed_dict={sample:X})
                    step += 1
                    if step % 10 == 0:
                        loss_cur = loss.eval(feed_dict={sample: X})
                        rel_imp = abs(loss_cur - loss_prev) / abs(loss_prev)
                        if rel_imp < 1e-2:
                            break
                        loss_prev = loss_cur
                loss_final = loss.eval(feed_dict={sample: X})
                if loss_final < best_of_runs:
                    updated = True
                    best_of_runs = loss_final
                    proj_mat = tf.concat([v, u], 1).eval()
                    dot_prod = tf.reduce_sum(tf.multiply(u, v)).eval()
        if not updated:
            self.log.info('WARNING: possible bug in the worst 2d projection')
        return proj_mat, dot_prod

    def train_epoch(self, sess, data_gen):
        iters = int(data_gen.train_size/self.config.trainer['batch_size'])
        
        losses = {
            'loss_wae' : [],
            'loss_rec' : [],
            'loss_match' : []
        }
        self.decay = 1.
        for it in range(iters):

            # Sample batches of data points and Pz noise

            batch_noise = self.sample_pz(self.config.trainer['batch_size'])
            batch = next(data_gen.next_train_batch(self.config.trainer['batch_size']))

            # Update encoder and decoder

            feed_d = {
                self.sample_points: batch['data'],
                self.sample_noise: batch_noise,
                self.lr_decay: self.decay,
                self.wae_lambda: self.config.model['lambda'],
                self.is_training: True}

            [_, loss, loss_rec, loss_match] = sess.run(
                [self.ae_opt,
                    self.wae_objective,
                    self.loss_reconstruct,
                    self.penalty],
                feed_dict=feed_d)

            # Update the adversary in Z space for WAE-GAN

            if self.config.model['z_test'] == 'gan':
                loss_adv = self.loss_gan[0]
                _ = sess.run(
                    [self.z_adv_opt, loss_adv],
                    feed_dict={self.sample_points: batch['data'],
                                self.sample_noise: batch_noise,
                                self.wae_lambda: self.config.model['lambda'],
                                self.lr_decay: self.decay,
                                self.is_training: True})

            losses['loss_wae'].append(loss)
            losses['loss_rec'].append(loss_rec)
            losses['loss_match'].append(loss_match)
            
        out_string = " ".join(["| "+key +" " + str(np.mean(losses[key])) for key in losses]) 
        self.log.info(out_string)

        return losses




    def add_sigmas_debug(self):

        # Ops to debug variances of random encoders
        enc_sigmas = self.enc_sigmas
        enc_sigmas = tf.Print(
            enc_sigmas,
            [tf.nn.top_k(tf.reshape(enc_sigmas, [-1]), 1).values[0]],
            'Maximal log sigmas:')
        enc_sigmas = tf.Print(
            enc_sigmas,
            [-tf.nn.top_k(tf.reshape(-enc_sigmas, [-1]), 1).values[0]],
            'Minimal log sigmas:')
        self.enc_sigmas = enc_sigmas

        enc_sigmas_t = tf.transpose(self.enc_sigmas)
        max_per_dim = tf.reshape(tf.nn.top_k(enc_sigmas_t, 1).values, [-1, 1])
        min_per_dim = tf.reshape(-tf.nn.top_k(-enc_sigmas_t, 1).values, [-1, 1])
        avg_per_dim = tf.reshape(tf.reduce_mean(enc_sigmas_t, 1), [-1, 1])
        per_dim = tf.concat([min_per_dim, max_per_dim, avg_per_dim], axis=1)
        self.debug_sigmas = per_dim
    
    def generate_curves(self, sess, noise):
        # generate random samples from the prior
        noise = noise[0]

        sample_gen = sess.run(self.decoded,
                        feed_dict={self.sample_noise: noise,
                                   self.is_training: False})
        return sample_gen

    def reconstruct_curves(self,sess, curves):

        [loss_rec_test, enc_test, rec_test] = sess.run(
                        [self.loss_reconstruct, self.encoded, self.reconstructed],
                        feed_dict={self.sample_points: curves,
                                   self.is_training: False})
        return {"rec_loss": loss_rec_test,
                "z_vectors": enc_test,
                "rec_curves": rec_test}

