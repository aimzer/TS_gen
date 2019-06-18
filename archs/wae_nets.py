import numpy as np
import tensorflow as tf
from ops.tf_ops import relu, lrelu, tanh, sig, bn, batch_norm, linear, conv1d, deconv1d

def encoder(config, inputs, is_training=False):

    if config.model['e_noise'] == 'add_noise':
        # Particular instance of the implicit random encoder
        def add_noise(x):
            shape = tf.shape(x)
            return x + tf.truncated_normal(shape, 0.0, 0.01)
        def do_nothing(x):
            return x
        inputs = tf.cond(is_training,
                         lambda: add_noise(inputs), lambda: do_nothing(inputs))
    

    if config.arch['e_arch'] == 'mlp':
        # Encoder uses only fully connected layers with ReLus
        
        num_filters = config.arch['e_num_filters']
        num_layers = config.arch['e_num_layers']

        hi = inputs

        layer_x = inputs
        for i in range(num_layers -1):
            layer_x = linear(config, layer_x, num_filers, scope='h%d_lin' % i)
            layer_x = relu(layer_x)
            if config.arch['e_batch_norm']:
                layer_x = batch_norm(config, layer_x, is_training, scope='h%d_bn' % i)
            
        logits = layer_x

    elif config.arch['e_arch'] == 'dcgan':
        # Fully convolutional architecture similar to DCGAN
        inputs = tf.expand_dims(inputs, -1)
        logits = dcgan_encoder_1D(config, inputs, is_training)

    else:
        raise ValueError('%s Unknown encoder architecture' % opts['e_arch'])

    if config.model['e_noise'] != 'gaussian':
        res = linear(config, logits, config.model['zdim'], scope='hfinal_lin')

    else:
        mean = linear(config, logits, config.model['zdim'], scope='mean_lin')
        log_sigmas = linear(opts, layer_x,
                                config.model['zdim'], scope='log_sigmas_lin')
        res = (mean, log_sigmas)

    noise_matrix = None

    if config.model['e_noise'] == 'implicit':
        # We already encoded the picture X -> res = E_1(X)
        # Now we return res + A(res) * eps, which is supposed
        # to project a noise on the directions depending on the
        # place in latent space
        sample_size = tf.shape(res)[0]
        eps = tf.random_normal((sample_size, config.model['zdim']),
                                0., 1., dtype=tf.float32)
        eps_mod, noise_matrix = transform_noise(config, res, eps)
        res = res + eps_mod

    if config.model['pz'] == 'sphere':
        # Projecting back to the sphere
        res = tf.nn.l2_normalize(res, dim=1)
    elif config.model['pz'] == 'uniform':
        # Mapping back to the [-1,1]^zdim box
        res = tf.nn.tanh(res)

    return res, noise_matrix

def decoder(config, noise, is_training=True):
    
    if config.arch['g_arch'] == 'mlp':
        # Architecture with only fully connected layers and ReLUs
        
        
        kernal_size = config.arch['g_conv_kernal']
        num_layers  = config.arch['g_num_layers']
        num_filers  = config.arch['g_num_filters']
        layer_x = noise
        for i in range(num_layers -1):
            layer_x = linear(config, layer_x, num_filers, scope='h%d_lin' % i)
            layer_x = tf.nn.relu(layer_x)
            if config.arch['g_batch_norm']:
                layer_x = batch_norm(config, layer_x, is_training, scope='h%d_bn' % i)

        out = linear(config, layer_x, config.model['input_curve_dim'], scope='h%d_lin' % (i+1))
        
        
    elif config.arch['g_arch'] == 'dcgan':
        # Fully convolutional architecture similar to DCGAN
        out = dcgan_decoder_1D(config, noise, is_training)
        
    else:
        raise ValueError('%s Unknown decoder architecture' % config.arch['g_arch'])

    return tf.nn.tanh(out), out


def dcgan_encoder_1D(config, inputs, is_training=False, reuse=False):


    num_layers  = config.arch['e_num_layers']
    num_filters = config.arch['e_num_filters']
    kernal_size = config.arch['e_conv_kernal']

    layer_x = inputs

    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)

        layer_x = conv1d(config, layer_x, num_filters / scale, conv_filters_dim=kernal_size, scope='h%d_conv' % i)
        layer_x = relu(layer_x)
        if config.arch['e_batch_norm']:
            layer_x = bn(layer_x, scope='h%d_bn' % i, is_training = is_training)
        
    return layer_x
    


def dcgan_decoder_1D(config, noise, is_training=False):

    num_filters  = config.arch['g_num_filters']
    num_layers   = config.arch['g_num_layers']
    kernal_size  = config.arch['g_conv_kernal']

    batch_size = tf.shape(noise)[0]

    height = int(config.model['input_curve_dim'] / 2**num_layers)

    layer_x = noise

    layer_x = linear(config, layer_x, num_filters * height, scope='h0_lin')
    layer_x = tf.reshape(layer_x, [batch_size, height, num_filters])
    layer_x = relu(layer_x)

    for i in range(num_layers):
        scale = 2**(i+1)
        
        _out_shape = [batch_size, int(height * scale), int(num_filters / scale)]
        layer_x = deconv1d(config, layer_x, output_shape=_out_shape,  conv_filters_dim = kernal_size, scope='h%d_deconv' % i)
        
        if config.arch['g_batch_norm']:
            layer_x = batch_norm(config, layer_x, is_training, scope='h%d_bn' % i)
        layer_x = relu(layer_x)

    _out_shape = [batch_size, config.model['input_curve_dim'], 1]
    res = conv1d(config, layer_x, 1, d_h=1, conv_filters_dim = kernal_size, scope='hfinal_deconv')
    
    res = tf.squeeze(res, -1)
    return res 


def z_adversary(config, inputs, reuse=False):
    num_units = config.arch['d_num_filters']
    num_layers = config.arch['d_num_layers']
    nowozin_trick = config.model['gan_p_trick']
    # No convolutions as GAN happens in the latent space
    hi = inputs
    for i in range(num_layers):
        hi = linear(config, hi, num_units, scope='h%d_lin' % (i + 1))
        hi = relu(hi)
    hi = linear(config, hi, 1, scope='hfinal_lin')
    if nowozin_trick:
        # We are doing GAN between our model Qz and the true Pz.
        # Imagine we know analytical form of the true Pz.
        # The optimal discriminator for D_JS(Pz, Qz) is given by:
        # Dopt(x) = log dPz(x) - log dQz(x)
        # And we know exactly dPz(x). So add log dPz(x) explicitly 
        # to the discriminator and let it learn only the remaining
        # dQz(x) term. This appeared in the AVB paper.
        assert config.model['pz'] == 'normal', \
            'The GAN Pz trick is currently available only for Gaussian Pz'
        sigma2_p = float(config.model['pz_scale']) ** 2
        normsq = tf.reduce_sum(tf.square(inputs), 1)
        hi = hi - normsq / 2. / sigma2_p \
                - 0.5 * tf.log(2. * np.pi) \
                - 0.5 * config.model['zdim'] * np.log(sigma2_p)
    return hi


def transform_noise(config, code, eps):
    hi = code
    T = 3
    for i in xrange(T):
        # num_units = max(opts['zdim'] ** 2 / 2 ** (T - i), 2)
        num_units = max(2 * (i + 1) * config.model['zdim'], 2)
        hi = ops.linear(config, hi, num_units, scope='eps_h%d_lin' % (i + 1))
        hi = tf.nn.tanh(hi)
    A = ops.linear(config, hi, config.model['zdim'] ** 2, scope='eps_hfinal_lin')
    A = tf.reshape(A, [-1, config.model['zdim'], config.model['zdim']])
    eps = tf.reshape(eps, [-1, 1, config.model['zdim']])
    res = tf.matmul(eps, A)
    res = tf.reshape(res, [-1, config.model['zdim']])
    return res, A
    # return ops.linear(opts, hi, opts['zdim'] ** 2, scope='eps_hfinal_lin')
