# ===========================
# -*- coding: utf-8 -*-
# Author: DSFNet Authors
# Date: 2024-09-27
# Keep Coding, Keep Thinking
# ===========================
import tensorflow as tf


def batch_norm(
        name,
        input_feature,
        eps=1e-05,
        momentum=0.9,
        rescale=True,
        is_training=tf.cast(True, tf.bool),
):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        params_shape = input_feature.get_shape()[1]
        # init beta and gamma
        if rescale:
            beta = tf.get_variable(
                name + '_beta',
                [params_shape],
                initializer=tf.zeros_initializer,
                trainable=True
            )
            gamma = tf.get_variable(
                name + '_gamma',
                [params_shape],
                initializer=tf.ones_initializer,
                trainable=True
            )
        else:
            beta = None
            gamma = None
        # init mean and var
        moving_mean = tf.get_variable(
            name + '_mean',
            [params_shape],
            initializer=tf.zeros_initializer,
            trainable=False
        )
        moving_variance = tf.get_variable(
            name + '_variance',
            [params_shape],
            initializer=tf.ones_initializer,
            trainable=False
        )

        def train():
            sample_mean, sample_variance = tf.nn.moments(input_feature, [0])

            running_mean = momentum * moving_mean + (1 - momentum) * sample_mean
            running_var = momentum * moving_variance + (1 - momentum) * sample_variance

            assign_mean = moving_mean.assign(running_mean)
            assign_variance = moving_variance.assign(running_var)
            with tf.control_dependencies([assign_mean, assign_variance]):
                batch_norm_value = tf.nn.batch_normalization(
                    input_feature,
                    sample_mean,
                    sample_variance,
                    beta,
                    gamma,
                    eps
                )
                return batch_norm_value, moving_mean, moving_variance

        def predict():
            batch_norm_value = tf.nn.batch_normalization(
                input_feature,
                moving_mean,
                moving_variance,
                beta,
                gamma,
                eps
            )
            return batch_norm_value, moving_mean, moving_variance

        return tf.cond(is_training, train, predict)


def batch_norm_moving(
        name,
        input_feature,
        eps=1e-05,
        momentum=0.999,
        rescale=True,
        is_training=tf.cast(True, tf.bool),
):
    with tf.variable_scope(name):
        params_shape = input_feature.get_shape()[1]

        if rescale:
            beta = tf.get_variable(
                name + '_beta',
                [params_shape],
                initializer=tf.zeros_initializer,
                trainable=True
            )
            gamma = tf.get_variable(
                name + '_gamma',
                [params_shape],
                initializer=tf.ones_initializer,
                trainable=True
            )
        else:
            beta = None
            gamma = None

        moving_mean = tf.get_variable(
            name + '_mean',
            [params_shape],
            initializer=tf.zeros_initializer,
            trainable=False
        )
        moving_variance = tf.get_variable(
            name + '_variance',
            [params_shape],
            initializer=tf.ones_initializer,
            trainable=False
        )

        def train():

            sample_mean, sample_variance = tf.nn.moments(input_feature, [0])

            running_mean = momentum * moving_mean + (1 - momentum) * sample_mean
            running_var = momentum * moving_variance + (1 - momentum) * sample_variance

            assign_mean = moving_mean.assign(running_mean)
            assign_variance = moving_variance.assign(running_var)
            with tf.control_dependencies([assign_mean, assign_variance]):
                batch_norm_value = tf.nn.batch_normalization(
                    input_feature,
                    moving_mean,
                    moving_variance,
                    beta,
                    gamma,
                    eps
                )
                return batch_norm_value, moving_mean, moving_variance

        def predict():
            batch_norm_value = tf.nn.batch_normalization(
                input_feature,
                moving_mean,
                moving_variance,
                beta,
                gamma,
                eps
            )
            return batch_norm_value, moving_mean, moving_variance

        return tf.cond(is_training, train, predict)
