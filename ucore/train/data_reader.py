# ===========================
# -*- coding: utf-8 -*-
# Author: DSFNet Authors
# Date: 2024-09-27
# Keep Coding, Keep Thinking
# ===========================

from __future__ import print_function, division

import tensorflow as tf

from ..config import dsfnet_configer


class DataReaderExpertV1:
    def __init__(self, op=None):
        self.model_w = ''
        self.model_b = ''
        self.train_iterator = ''
        self.eval_iterator = ''
        self.specific_iterator = ''
        self.train_data_saver = None
        self.eval_data_saver = None
        # read data from sql table
        if op is not None:
            self.train_iterator = self.data_to_streaming(op.read_train_table, input_path=dsfnet_configer.train_path)
            self.eval_iterator = self.data_to_streaming(op.read_valid_table, input_path=dsfnet_configer.test_path,
                                                        read_epochs=1)

    def decode_table_record(self, *items):
        route_feature = tf.strings.to_number(
            tf.reshape(tf.strings.split(
                tf.strings.regex_replace(items[0], "[\\[\\]\\s\"\']", ""), sep=',', result_type='RaggedTensor'), [-1]),
            tf.float32)
        main_diff = tf.strings.to_number(
            tf.reshape(tf.strings.split(
                tf.strings.regex_replace(items[3], "[\\[\\]\\s\"\']", ""), sep=',', result_type='RaggedTensor'), [-1]),
            tf.float32)
        context_feature = tf.strings.to_number(
            tf.reshape(tf.strings.split(
                tf.strings.regex_replace(items[4], "[\\[\\]\\s\"\']", ""), sep=',', result_type='RaggedTensor'), [-1]),
            tf.float32)
        strategy_feature = tf.strings.to_number(
            tf.reshape(tf.strings.split(
                tf.strings.regex_replace(items[5], "[\\[\\]\\s\"\']", ""), sep=',', result_type='RaggedTensor'), [-1]),
            tf.float32)

        label = tf.cast(tf.reshape(items[6], [-1]), tf.float32)
        coverage = tf.strings.to_number(
            tf.reshape(tf.strings.split(
                tf.strings.regex_replace(items[7], "[\\[\\]\\s\"\']", ""), sep=',', result_type='RaggedTensor'), [-1]),
            tf.float32)
        return route_feature, main_diff, context_feature, strategy_feature, label, coverage

    def padding(self, dataset, batch_num, drop_remainder):
        dataset = dataset.padded_batch(
            batch_num,
            padded_shapes=(
                [None],
                [None],
                [None],
                [None],
                [None],
                [None],
            ),
            padding_values=(
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
            ),
            drop_remainder=drop_remainder
        )
        return dataset

    def data_to_streaming(self, reader, input_path=None, read_epochs=None):
        print("data table path: {}".format(input_path))
        if not dsfnet_configer.LOCAL_TRAIN:
            data_set = tf.data.TableRecordDataset(
                input_path[0],
                record_defaults=('', '', '', '', 0.0, ''),
                selected_cols="route_feature,main_diff_feature,context_feature,strategy_feature,label,coverage",
                num_threads=dsfnet_configer.TABLE_PARALLEL_SIZE
            )
        else:
            data_set = reader(
                ('', '', '', '', 0.0, ''),
                selected_cols="route_feature,main_diff_feature,context_feature,strategy_feature,label,coverage",
                num_threads=dsfnet_configer.TABLE_PARALLEL_SIZE
            )

        data_set = data_set.map(
            self.decode_table_record,
            num_parallel_calls=dsfnet_configer.DECODE_PARALLEL_SIZE
        )
        if read_epochs:
            data_set = data_set.repeat(read_epochs)
        else:
            data_set = data_set.repeat(dsfnet_configer.read_epochs)
        data_set = data_set.shuffle(dsfnet_configer.shuffle_size)
        data_set = self.padding(
            data_set, dsfnet_configer.batch_size, True)
        data_set = data_set.prefetch(dsfnet_configer.PREFETCH_NUM)
        iterator = data_set.make_initializable_iterator()
        return iterator
