import tensorflow as tf

from tqdm import tqdm
import numpy as np

from base.base_model import BaseModel
from archs.nets import generator_net, discriminator_net
from ops.tf_ops import average_gradients

class GAN(BaseModel):
    def __init__(self, config):
        super(GAN, self).__init__(config)
        
        with tf.device('/cpu:0'):
            self.add_inputs_placeholders()
            self.add_training_placeholders()
            
            self.build_model()
            self.add_optimizers()

            self.add_eval_ops()
            
            self.add_savers()        

            self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def add_inputs_placeholders(self):

        input_curve_dim = self.config.model['input_curve_dim']
        input_noise_dim = self.config.model['input_noise_dim']

        self.real_curves = tf.placeholder(tf.float32, shape=[None] + [input_curve_dim], name='real_images')
        shape = [None, input_noise_dim]
        self.noise       = tf.placeholder(tf.float32, shape=shape, name='noise')
        if 'label' in self.config.data:
            input_label_dim = self.config.model['input_label_dim']
            self.label   = tf.placeholder(tf.float32, shape=[None] + [input_label_dim], name='noise')
        else:
            self.label = None

        if 'gpus' in self.config.trainer and len(self.config.trainer['gpus'])>1:
            self.real_curves_splits = tf.split(self.real_curves, len(self.config.trainer['gpus']), axis=0)
            self.noise_splits       = tf.split(self.noise      , len(self.config.trainer['gpus']), axis=0)
            if 'label' in self.config.data:
                self.label_splits   = tf.split(self.label      , len(self.config.trainer['gpus']), axis=0)
        else:
            self.real_curves_splits = [self.real_curves]
            self.noise_splits       = [self.noise]
            if 'label' in self.config.data:
                self.label_splits   = [self.label]


    def add_training_placeholders(self):
        
        self.is_training = tf.placeholder(tf.bool)

    def build_model(self):
        devices = ['/gpu:{}'.format(i) for i in self.config.trainer['gpus']]
        
        self.log.info(devices)
        self.errDs_real  = []
        self.errDs_fake  = []
        self.errDs       = []
        self.errGs       = []

        if self.config.model['loss'] == 'wgan':
            self.gradient_penaltys = []
        
        reuse = False
        self.gen_curves_gpu =[]
        for i, device in enumerate(devices):
            with tf.device(device):
                label_splits_i = None
                if 'label' in self.config.data:
                    label_splits_i = self.label_splits[i]
                with tf.variable_scope("generator", reuse=reuse):
                    gen_curves = generator_net    (self.config, [self.noise_splits[i],      label_splits_i], self.is_training)
                with tf.variable_scope("discriminator", reuse=reuse):
                    errD_real  = discriminator_net(self.config, [self.real_curves_splits[i],label_splits_i], self.is_training)
                with tf.variable_scope("discriminator", reuse=True):
                    errD_fake  = discriminator_net(self.config, [gen_curves,                label_splits_i], self.is_training)

                assert self.config.model['loss'] in ['gan','lsgan','wgan'], "Unknown model type "+ self.config.model['loss']
                self.gen_curves_gpu.append(gen_curves)
                e = 1e-12
                
                if self.config.model['loss'] == 'gan':
                    errD_real = tf.nn.sigmoid (errD_real)
                    errD_fake = tf.nn.sigmoid (errD_fake)
                    errG      = tf.reduce_mean(-tf.log(errD_fake + e))
                    errD      = tf.reduce_mean(-(tf.log(errD_real+e)+tf.log(1-errD_fake+e)))

                elif self.config.model['loss'] == 'lsgan':
                    errD_real = tf.nn.sigmoid(errD_real)
                    errD_fake = tf.nn.sigmoid(errD_fake)
                    errD      = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
                    errG      = tf.reduce_mean(0.5*(tf.square(errD_fake - 1)))

                elif self.config.model['loss'] == 'wgan':
                    # cost functions
                    errD      = tf.reduce_mean(errD_real) - tf.reduce_mean(errD_fake)
                    errG      = tf.reduce_mean(errD_fake)

                    # gradient penalty
                    epsilon   = tf.random_uniform([], 0.0, 1.0)
                    x_hat     = self.real_curves_splits[i]*epsilon + (1-epsilon)*gen_curves
                    
                    with tf.variable_scope("discriminator", reuse=True):
                        d_hat = discriminator_net(self.config, [x_hat,label_splits_i], self.is_training)

                    gradients = tf.gradients(d_hat, x_hat)[0]
                    slopes    = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
                    errD      += gradient_penalty
                    self.gradient_penaltys.append(gradient_penalty)
                
                self.errGs.append(errG)
                self.errDs.append(errD)
                self.errDs_fake.append(errD_real)
                self.errDs_real.append(errD_real)

                reuse = True
        
        if self.config.model['loss'] == 'wgan':
            self.gradient_penalty = tf.add_n(self.gradient_penaltys) / len(devices)
        self.errG = tf.add_n(self.errGs) / len(devices)
        self.errD = tf.add_n(self.errDs) / len(devices)
        self.errD_fake = tf.reduce_mean(tf.add_n(self.errDs_fake)) / len(devices)
        self.errD_real = tf.reduce_mean(tf.add_n(self.errDs_real)) / len(devices)

        self.gen_curves_gpu = tf.concat(self.gen_curves_gpu, 0)

        #self.gradient_penalty = gradient_penalty

    def add_optimizers(self):

        generator_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        devices = ['/gpu:{}'.format(i) for i in self.config.trainer['gpus']]
        
        G_grads    = []
        D_grads    = []

        if self.config.model['loss'] == 'wgan':
            self.gp_grads = []
        
        for i, device in enumerate(devices):
            with tf.device(device):
                lr    = self.config.model['optim_lr']
                beta1 = self.config.model['optim_beta1']
                beta2 = self.config.model['optim_beta2']

                G_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2)
                D_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2)

                G_grads.append(G_train_op.compute_gradients(self.errGs[i], var_list = generator_vars))
                D_grads.append(D_train_op.compute_gradients(self.errDs[i], var_list = discriminator_vars))

        G_grads = average_gradients(G_grads)
        D_grads = average_gradients(D_grads)

        self.G_train_op = G_train_op.apply_gradients(G_grads, global_step=self.global_step)
        self.D_train_op = D_train_op.apply_gradients(D_grads)

    def add_eval_ops(self):
        
        if 'label' in self.config.data:
            label = self.label
        else:
            label = None
        with tf.variable_scope("generator", reuse=True):
            self.gen_curves = generator_net    (self.config, [self.noise,       label], self.is_training)
        with tf.variable_scope("discriminator", reuse=True):
            self.disc_out  = discriminator_net(self.config, [self.real_curves, label], self.is_training)

    def add_savers(self):
        
        saver = tf.train.Saver(max_to_keep = self.config.model['max_to_keep'])

        tf.add_to_collection('real_curves_ph', self.real_curves)
        tf.add_to_collection('noise_ph',       self.noise)
        if 'label' in self.config.data:
            tf.add_to_collection('label_ph',   self.label)
        tf.add_to_collection('is_training_ph', self.is_training)
        
        tf.add_to_collection('generated',    self.gen_curves)
        tf.add_to_collection('discriminated',    self.disc_out)
        tf.add_to_collection('D_error_fake', self.errD_fake)
        tf.add_to_collection('D_error_real', self.errD_real)
        if(self.config.model['loss']=='wgan'):
            tf.add_to_collection('gradient_penalty_ph', self.gradient_penalty)

        tf.add_to_collection('G_train_op', self.G_train_op)
        tf.add_to_collection('D_train_op', self.D_train_op)

        self.saver = saver

    def train_epoch(self, sess, data_gen):
        iters = int(data_gen.train_size/self.config.trainer['discriminator_iters']/self.config.trainer['batch_size'])
        #loop = tqdm(range(iters))
        
        data_gen.reset_counter()
        loss_D_real = []
        loss_D_fake = []
        loss_D = []
        loss_G = []
        losses = {
            'loss_D_real' : [],
            'loss_D_fake' : [],
            'loss_D' : [],
            'loss_G' : []
        }
        for i in range(iters):
            for i in range(self.config.trainer['discriminator_iters']):
                
                noise = self.sample_noise()
                batch = next(data_gen.next_train_batch(self.config.trainer['batch_size']))
                real_curves = batch['data']
                real_labels = batch['labels']

                feed_dict={self.real_curves : real_curves, self.noise : noise, self.is_training : True}
                if 'label' in self.config.data:
                    feed_dict[self.label]=real_labels
                
                sess.run(self.D_train_op, feed_dict=feed_dict)

            noise = np.random.normal(0,1,[self.config.trainer['batch_size'], self.config.model['input_noise_dim']])
            
            feed_dict={self.real_curves : real_curves, self.noise : noise, self.is_training : True}
            if 'label' in self.config.data:
                feed_dict[self.label]=real_labels

            out_list = [self.G_train_op,self.errD_real, self.errD_fake, self.errD, self.errG]
            
            out_list = sess.run(out_list, feed_dict=feed_dict)
            
            losses['loss_D_real'].append(np.mean(out_list[1]))
            losses['loss_D_fake'].append(np.mean(out_list[2]))
            losses['loss_D'].append(out_list[3])
            losses['loss_G'].append(out_list[4])
            if self.config.trainer['verbose']:
                loop_string = "[G : %+06.3f] [D : %+06.3f] [Df : %+06.3f] [Dr : %+06.3f]" % (out_list[4], out_list[3], np.mean(out_list[2]), np.mean(out_list[1]))
                loop.set_description(loop_string)
            
        self.log.info("D : "+ str(np.mean(losses['loss_D']))+ 
                   " | G : "+str(np.mean(losses['loss_G']))+ 
                   " | DR : "+str(np.mean(losses['loss_D_real'])) +
                   " | DF : "+ str(np.mean(losses['loss_D_fake'])))

        return losses
            
    def sample_noise(self, N=None):
        if N is None:
            N=self.config.trainer['batch_size']
        noise = np.random.normal(0,1,[N, self.config.model['input_noise_dim']])
        return noise

            

    def generate_curves(self, sess, generator_input):
        noise_input = generator_input[0]
        label_input = generator_input[1]
        if 'label' in self.config.data:
            feed_dict={self.noise : noise_input, self.label : label_input, self.is_training:False}
        else:
            feed_dict={self.noise : noise_input, self.is_training:False}
        gen_curves = sess.run(self.gen_curves,\
                            feed_dict=feed_dict)

        return gen_curves

    def generate_curves_gpu(self, sess, generator_input):
        noise_input = generator_input[0]
        label_input = generator_input[1]
        if 'label' in self.config.data:
            feed_dict={self.noise : noise_input, self.label : label_input, self.is_training:False}
        else:
            feed_dict={self.noise : noise_input, self.is_training:False}
        gen_curves = sess.run(self.gen_curves_gpu,\
                            feed_dict=feed_dict)
        return gen_curves


    def discriminator_output(self, sess, discriminator_input):
        curve_input = discriminator_input[0]
        label_input = discriminator_input[1]
        if label_input:
            feed_dict={self.real_curves : curve_input, self.label : label_input, self.is_training:False}
        else:
            feed_dict={self.real_curves : discriminator_input, self.is_training:False}
        disc_output = sess.run(self.disc_out,\
                            feed_dict=feed_dict)
        
        return disc_output

            
