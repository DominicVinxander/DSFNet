# ===========================
# -*- coding: utf-8 -*-
# Author: DSFNet Authors
# Date: 2024-09-27
# Keep Coding, Keep Thinking
# ===========================

import abc

import tensorflow as tf

from ..config import detour_configer


class ModelBase(object):
    def __init__(self, name, config, n_domain=7):
        print("tf.__version__=", tf.__version__)
        self.name = name
        self.config = config
        self.main_process()
        self.model_saver = tf.train.Saver(max_to_keep=1000)
        self.n_domain = n_domain

    def main_process(self, global_step=0):
        self._initial_net()
        self._add_summary()
        self.count_param()

    @abc.abstractmethod
    def _initial_net(self):
        raise NotImplementedError("Must Implement this method")

    @abc.abstractmethod
    def _add_summary(self):
        raise NotImplementedError("Must Implement this method")

    def save_model(self, sess, step):
        self.model_saver.save(sess, '{}/step_{}'.format(self.config.model_path, step), global_step=step)

    def load_model(self, sess):
        if self.config.use_type == 1 and len(self.config.global_step) != 0:
            file_name = '{}/step_{}-{}'.format(self.config.model_path, self.config.global_step, self.config.global_step)
            self.model_saver.restore(sess, file_name)
            print('Model restored from file: {}'.format(file_name))
            return True
        return False

    def count_param(self):  # Compute Network parameter
        total_parameters = 0
        for v in tf.trainable_variables():
            shape = v.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Network parameter: ', total_parameters)


class ModelBaseExporter(object):
    def __init__(self, name, config):
        self.name = "ModelBaseExporter"
        self.config = config
        self.main_process()

    def main_process(self):
        self._initial_net()

    @abc.abstractmethod
    def _initial_net(self):
        raise NotImplementedError("Must Implement this method")
