# ===========================
# -*- coding: utf-8 -*-
# Author: DSFNet Authors
# Date: 2024-09-27
# Keep Coding, Keep Thinking
# ===========================

import numpy as np
import tensorflow as tf

from . import batch_norm


def layer_norm(name, inputs):
    outputs = tf.contrib.layers.layer_norm(
        inputs,
        scope=name
    )
    return outputs


def neuron_layer(
        name,
        input_feature,
        n_neurons,
        activation=None,
        if_batch_norm=False,
        if_layer_norm=False,
        is_training=tf.cast(True, tf.bool),
        if_bias=True,
        input_shape=[]
):
    with tf.name_scope(name):
        n_inputs = int(input_feature.get_shape()[-1])
        if len(input_shape) > 2:
            input_feature = tf.reshape(input_feature, [-1, n_inputs])
        if activation == 'tanh':
            initializer = tf.initializers.GlorotUniform()
            weights = tf.Variable(initializer(shape=[n_inputs, n_neurons]))
        else:
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.random_normal((n_inputs, n_neurons), stddev=stddev)
            weights = tf.Variable(init, name='weights')
        bias = tf.Variable(tf.zeros([n_neurons]), name='biase')
        if if_bias:
            out_value = tf.matmul(input_feature, weights) + bias
        else:
            out_value = tf.matmul(input_feature, weights)
        # batch_normalization
        if if_batch_norm:
            out_value, mean, var = batch_norm.batch_norm(
                '{}_batch_norm'.format(name),
                out_value,
                is_training=is_training
            )
        # layer_norm
        if if_layer_norm:
            out_value = layer_norm(
                '{}_layer_norm'.format(name),
                out_value
            )
        # activation funs
        if activation == 'relu':
            out_value = tf.nn.relu(out_value)
        elif activation == 'tanh':
            out_value = tf.nn.tanh(out_value)
        elif activation == 'leaky_relu':
            out_value = tf.nn.leaky_relu(out_value)
        elif activation == 'elu':
            out_value = tf.nn.elu(out_value)
        elif activation == 'sigmoid':
            out_value = tf.nn.sigmoid(out_value)
        if len(input_shape) > 2:
            out_value = tf.reshape(out_value, input_shape[:-1] + [n_neurons])
        return out_value, weights


# star neuron_layer
def neuron_layer_partitioned(
        name,
        input_feature,
        n_neurons,
        activation=None,
        if_batch_norm=False,
        if_layer_norm=False,
        is_training=tf.cast(True, tf.bool),
        if_bias=True,
        input_shape=[],
        n_domain=7,
        cur_domain_id=1,
        base_domain_id=0,
        n_inputs=288,
        strategy_feature=None,
        embedding_weight=None
        # mask=None
):
    with tf.name_scope(name):
        if len(input_shape) > 2:
            input_feature = tf.reshape(input_feature, [-1, n_inputs])
        if activation == 'tanh':
            initializer = tf.initializers.GlorotUniform()
            weights = tf.Variable(initializer(shape=[n_inputs, n_neurons]))
        else:
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.random_normal((n_domain + 1, n_inputs, n_neurons), stddev=stddev)
            weights = tf.Variable(init, name='weights')  # (#e+1, d_i,d_o)
        bias = tf.Variable(tf.zeros([n_domain + 1, n_neurons]), name='biase')

        if cur_domain_id != -1:
            domain_weights = tf.multiply(tf.gather(weights, base_domain_id), tf.gather(weights, cur_domain_id))
            if if_bias:
                domain_bias = tf.add(tf.gather(bias, base_domain_id), tf.gather(bias, cur_domain_id))
                out_value = tf.matmul(input_feature, domain_weights) + domain_bias
            else:
                out_value = tf.matmul(input_feature, domain_weights)

        else:
            if embedding_weight is None:
                print("embedding_weight error!")
            if strategy_feature is None:
                ori_strategy_weight = tf.sigmoid(embedding_weight) * 2
            else:
                strategy_feature = tf.cast(strategy_feature, dtype=tf.int32)
                strategy_feature = tf.squeeze(strategy_feature, axis=1)  # [B,]
                ori_strategy_weight = tf.nn.softmax(
                    tf.nn.embedding_lookup(embedding_weight, strategy_feature))  # [B, expert_num]
            strategy_weight = tf.expand_dims(tf.expand_dims(ori_strategy_weight, axis=2), axis=3)  # (B,#e,1,1)
            domain_weights = tf.reduce_sum(tf.multiply(strategy_weight, weights[1:n_domain + 1]), axis=1)  # (B,d_i,d_o)
            input_feature = tf.expand_dims(input_feature, axis=1)  # (B,1,d_i)
            if if_bias:
                strategy_weight_b = tf.expand_dims(ori_strategy_weight, axis=2)
                domain_bias = tf.expand_dims(
                    tf.reduce_sum(tf.multiply(strategy_weight_b, bias[1:n_domain + 1]), axis=1), axis=1)
                out_value = tf.matmul(input_feature, domain_weights) + domain_bias
            else:
                out_value = tf.matmul(input_feature, domain_weights)
            out_value = tf.squeeze(out_value, axis=1)
        # batch_normalization
        if if_batch_norm:
            out_value, mean, var = batch_norm.batch_norm(
                '{}_batch_norm'.format(name),
                out_value,
                is_training=is_training
            )
        # layer_norm
        if if_layer_norm:
            out_value = layer_norm(
                '{}_layer_norm'.format(name),
                out_value
            )
        # activation funs
        if activation == 'relu':
            out_value = tf.nn.relu(out_value)
        elif activation == 'tanh':
            out_value = tf.nn.tanh(out_value)
        elif activation == 'leaky_relu':
            out_value = tf.nn.leaky_relu(out_value)
        elif activation == 'elu':
            out_value = tf.nn.elu(out_value)
        if len(input_shape) > 2:
            out_value = tf.reshape(out_value, input_shape[:-1] + [n_neurons])
        return out_value, weights
