
import tensorflow as tf
from ops.tf_ops import relu, lrelu, tanh, sig, bn, linear, conv1d, deconv1d

def sequence_length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

def discriminator_net(config, input_, is_training=False):

    if config.arch['d_arch'] == 'mlp':
        
        if 'label' in config.data:
            inputs_ = tf.concat(input_, -1)
        else:
            inputs_ = input_[0]
        kernal_size = config.arch['d_conv_kernal']
        num_layers  = config.arch['d_num_layers']
        num_filers  = config.arch['d_num_filters']
        layer_x = inputs_
        for i in range(num_layers -1):
            layer_x = linear(config, layer_x, num_filers, scope='h%d_lin' % i)
            if config.arch['d_batch_norm']:
                layer_x = bn(layer_x, scope='h%d_bn' % i, is_training = is_training)
            layer_x = lrelu(layer_x)

        res = linear(config, layer_x, 1, scope='h%d_lin' % (i+1))
    
    elif config.arch['d_arch'] == 'dcgan':
        # Fully convolutional architecture similar to DCGAN
        if 'label' in config.data:
        
            inputs_ = tf.expand_dims(input_[0], -1)
            labels_ = tf.expand_dims(input_[1], 1)

            inputs_shapes = inputs_.shape.as_list()
            #labels_shapes = labels_.shape.as_list()
            
            labels_ = tf.tile(labels_, [1, inputs_shapes[1], 1])
            inputs_ = tf.concat([inputs_, labels_], -1)
        else:
            inputs_, _ = input_
            inputs_ = tf.expand_dims(inputs_, -1)

        res = dcgan_discriminator(config, inputs_, is_training)

    else:
        raise ValueError('%s Unknown encoder architecture' % config.arch['d_arch'])

    return res

def dcgan_discriminator(config, input_, is_training=False):
    num_layers  = config.arch['d_num_layers']
    num_filters  = config.arch['d_num_filters']
    kernal_size = config.arch['d_conv_kernal']

    layer_x = input_

    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)

        layer_x = conv1d(config, layer_x, num_filters / scale, conv_filters_dim=kernal_size, scope='h%d_conv' % i)
        #layer_x = tf.nn.layers.conv1d(layer_x,num_outputs=num_filters / scale,\
        #                        kernel_size=kernal_size,stride=2, name='h%d_conv' % i)
            

        if config.arch['d_batch_norm']:
            layer_x = bn(layer_x, scope='h%d_bn' % i, is_training = is_training)
        layer_x = lrelu(layer_x)

    #tf.reduce_mean(output, axis=[2])

    res = linear(config, layer_x, 1, scope='hfinal_lin')
    return res

def generator_net(config, input_, is_training=False):

    if config.arch['g_arch'] == 'mlp':
        if 'label' in config.data:
            input_ = tf.concat(input_, axis=1)
        else:
            input_ = input_[0]
        layer_x = input_
        num_filers   = config.arch['g_num_filters']
        num_layers   = config.arch['g_num_layers']
        for i in range(num_layers-1):
            layer_x = linear(config, layer_x, num_filers, scope='h%d_lin' % i)
            layer_x = tf.nn.relu(layer_x)
            if config.arch['g_batch_norm']:
                layer_x = bn(layer_x, scope='h%d_bn' % i, is_training = is_training)
                
        res = linear(config, layer_x, config.model['input_curve_dim'], scope='h%d_lin' % (i+1))
        res = tf.reshape(res, [-1] + [config.model['input_curve_dim']])

    elif config.arch['g_arch'] == 'dcgan':

        if 'label' in config.data:
            input_ = tf.concat(input_, axis=1)
        else:
            input_= input_[0]

        res = dcgan_generator(config, input_, is_training)
        res = tf.squeeze(res, -1)

    else:
        raise ValueError('%s Unknown generator architecture' % config.arch['g_arch'])
    
    
    return tanh(res)

def dcgan_generator(config, input_, is_training=False):
    
    num_filters  = config.arch['g_num_filters']
    num_layers   = config.arch['g_num_layers']
    kernal_size  = config.arch['g_conv_kernal']

    batch_size = tf.shape(input_)[0]

    height = int(config.model['input_curve_dim'] / 2**num_layers)

    layer_x = input_

    layer_x = linear(config, layer_x, num_filters * height, scope='h0_lin')
    layer_x = tf.reshape(layer_x, [batch_size, height, num_filters])
    layer_x = relu(layer_x)

    for i in range(num_layers):
        scale = 2**(i+1)
        
        _out_shape = [batch_size, int(height * scale), int(num_filters / scale)]
        layer_x = deconv1d(config, layer_x, output_shape=_out_shape,  conv_filters_dim = kernal_size, scope='h%d_deconv' % i)
        
        if config.arch['g_batch_norm']:
            layer_x = bn(layer_x, scope='h%d_bn' % i, is_training = is_training)
        layer_x = relu(layer_x)

    _out_shape = [batch_size, config.model['input_curve_dim'], 1]
    res = conv1d(config, layer_x, 1, d_h=1, conv_filters_dim = kernal_size, scope='hfinal_deconv')

    return res 

