import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers 
from tensorflow.keras import regularizers


def cbam_block(input_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    attention_feature = channel_attention(input_feature, ratio)
    attention_feature = spatial_attention(attention_feature)

    return attention_feature

def channel_attention(input_feature, ratio=16):
  
    #kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    kernel_initializer = tf.keras.initializers.VarianceScaling()
    bias_initializer = tf.constant_initializer(value=0.0)
  
    channel = input_feature.get_shape()[-1]
    avg_pool = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
        
    #assert avg_pool.get_shape()[1:] == (1,1,channel)
    avg_pool = layers.Dense(units=channel//ratio,
                                    activation=tf.nn.relu,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer)(avg_pool) 
    #assert avg_pool.get_shape()[1:] == (1,1,channel//ratio)
    avg_pool = layers.Dense(units=channel,
                                    activation=tf.nn.relu,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer)(avg_pool)    
    #assert avg_pool.get_shape()[1:] == (1,1,channel)

    max_pool = tf.reduce_max(input_feature, axis=[1,2], keepdims=True)    
    #assert max_pool.get_shape()[1:] == (1,1,channel)
    max_pool = layers.Dense(units=channel//ratio, activation=tf.nn.relu)(max_pool)   
    #assert max_pool.get_shape()[1:] == (1,1,channel//ratio)
    max_pool = layers.Dense(units=channel, activation=tf.nn.relu)(max_pool)   
    #assert max_pool.get_shape()[1:] == (1,1,channel)

    scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
    
    return input_feature * scale

def spatial_attention(input_feature):
    kernel_size = 7
    kernel_initializer = tf.keras.initializers.VarianceScaling()
    avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
    #assert avg_pool.get_shape()[-1] == 1
    max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
    #assert max_pool.get_shape()[-1] == 1
    concat = tf.concat([avg_pool,max_pool], 3)
    #assert concat.get_shape()[-1] == 2

    concat = layers.Conv2D(filters=1,
                                kernel_size=[kernel_size,kernel_size],
                                strides=[1,1],
                                padding="same",
                                activation=None,
                                kernel_initializer=kernel_initializer,
                                use_bias=False)(concat)
    #assert concat.get_shape()[-1] == 1
    concat = tf.sigmoid(concat, 'sigmoid')
    
    return input_feature * concat
    
    