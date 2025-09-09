# ===========================
# -*- coding: utf-8 -*-
# Author: DSFNet Authors
# Date: 2024-09-27
# Keep Coding, Keep Thinking
# ===========================

import tensorflow as tf

from ..config import detour_configer

tf.logging.set_verbosity(tf.logging.INFO)


def embed_user(user_feature):
    with tf.variable_scope("embed_user"):
        # sex
        sex_type = tf.cast(user_feature[:, detour_configer.SEX_FEAT_INDEX], dtype=tf.int32)
        sex_embedding = tf.get_variable(name='sex_embedding',
                                        shape=(detour_configer.SEX_NUM, detour_configer.SEX_EMBED_DIM))
        sex_feat = tf.nn.embedding_lookup(sex_embedding, sex_type)
        # age
        age_type = tf.cast(user_feature[:, detour_configer.AGE_FEAT_INDEX], dtype=tf.int32)
        age_embedding = tf.get_variable(name='age_embedding',
                                        shape=(detour_configer.AGE_NUM, detour_configer.AGE_EMBED_DIM))
        age_feat = tf.nn.embedding_lookup(age_embedding, age_type)
        # job
        j_type = tf.cast(user_feature[:, detour_configer.JOB_FEAT_INDEX], dtype=tf.int32)
        job_embedding = tf.get_variable(name='job_embedding',
                                        shape=(detour_configer.JOB_NUM, detour_configer.JOB_EMBED_DIM))
        job_feat = tf.nn.embedding_lookup(job_embedding, j_type)
        user_feature = tf.concat([sex_feat, age_feat, job_feat], axis=-1)
        print("User embedding done: self.user_feature.shape =", user_feature.shape)
        return user_feature


def embed_strategy(strategy_feature):
    with tf.variable_scope("embed_strategy"):
        poi_type = tf.cast(strategy_feature[:, detour_configer.POI_FEAT_INDEX], dtype=tf.int32)
        poi_embedding = tf.get_variable(name='poi_embedding',
                                        shape=(detour_configer.POI_NUM, detour_configer.POI_EMBED_DIM))
        poi_feat = tf.nn.embedding_lookup(poi_embedding, poi_type)
        day_type = strategy_feature[:, detour_configer.DAY_FEAT_INDEX]
        day_feat = tf.tile(tf.expand_dims(day_type, axis=-1), tf.constant([1, detour_configer.DAY_EMBED_DIM], tf.int32))
        if detour_configer.BASELINE_COMPARISON:
            raise NotImplementedError
        else:
            or_type = tf.cast(strategy_feature[:, detour_configer.OR_FEAT_INDEX], dtype=tf.int32)
            or_embedding = tf.get_variable(name='or_embedding',
                                           shape=(detour_configer.ODR_NUM, detour_configer.OR_EMBED_DIM))
            or_feat = tf.nn.embedding_lookup(or_embedding, or_type)
            dr_type = tf.cast(strategy_feature[:, detour_configer.DR_FEAT_INDEX], dtype=tf.int32)
            dr_embedding = tf.get_variable(name='dr_embedding',
                                           shape=(detour_configer.ODR_NUM, detour_configer.DR_EMBED_DIM))
            dr_feat = tf.nn.embedding_lookup(dr_embedding, dr_type)
            strategy_feature = tf.concat([strategy_feature[:, :detour_configer.POI_FEAT_INDEX],
                                          poi_feat, day_feat, or_feat, dr_feat,
                                          strategy_feature[:, detour_configer.DR_FEAT_INDEX + 1:]],
                                         axis=-1)  # (n,26+15)
        print("Strategy embedding done: self.strategy_feature.shape =", strategy_feature.shape)
        return strategy_feature
