# ===========================
# -*- coding: utf-8 -*-
# Author: DSFNet Authors
# Date: 2024-09-27
# Keep Coding, Keep Thinking
# ===========================

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from .Embedding import embed_strategy, embed_user
from .ModelBase import ModelBase
from .. import layers
from ..config import dsfnet_configer

tf.logging.set_verbosity(tf.logging.INFO)


class DSFNet(ModelBase):
    def __init__(self, config, name="ModelStar_rsdf"):
        super(DSFNet, self).__init__(name, config)

    def _add_summary(self):
        with tf.name_scope('cost'):
            if isinstance(self.cost, dict):
                tf.summary.scalar('cost', self.cost['loss'])
            else:
                tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('auc', self.auc)
            tf.summary.scalar('learning_rate', self.learning_rate)
        self.merged_summary = tf.summary.merge_all()

    def _initial_net(self, global_step=0):
        # ------------------------------------ step 1:  set placeholder  ------------------------------------
        self.expert_num = dsfnet_configer.DSFNET_EXPERT_NUM
        self.n_domain = self.expert_num
        self.expert_neuron0, self.expert_neuron1, self.expert_neuron2 = dsfnet_configer.EXPERT_NEURONS
        self.dynamic_neuron0, self.dynamic_neuron1 = dsfnet_configer.DYNAMIC_NEURONS
        self.route_feature = tf.placeholder(
            "float",
            [None, dsfnet_configer.route_feature_num],
            name="route_feature"
        )
        self.main_diff = tf.placeholder(
            "float",
            [None, dsfnet_configer.main_diff_feature_num],
            name="main_diff"
        )
        self.context_feature = tf.placeholder(
            'float',
            [None, dsfnet_configer.context_feature_num],
            name="context_feature"
        )

        self.strategy_feature = tf.placeholder(
            'float',
            [None, dsfnet_configer.strategy_feature_num],
            name="strategy_feature"
        )
        self.label = tf.placeholder(
            "float",
            [None, dsfnet_configer.label_num],
            name="label"
        )
        self.coverage = tf.placeholder(
            "float",
            [None, 1],
            name="coverage"
        )
        self.is_training = tf.placeholder(tf.bool)
        self.strategy_id = self.strategy_feature[:, 0]

        # -------------------------- step 2:  feature process before input the model  --------------------------
        if dsfnet_configer.USE_USER_EMBEDDING:
            self.user_feature = self.strategy_feature[:, dsfnet_configer.SCENARIO_FEAT_FUM:]
            self.strategy_feature_base = self.strategy_feature[:, 1:dsfnet_configer.SCENARIO_FEAT_FUM]
            self.user_feature = embed_user(self.user_feature)  # (n,6)
            self.strategy_feature_base = embed_strategy(self.strategy_feature_base)  # (n,26 or 41)
        else:
            raise NotImplementedError

        user_feature = self.user_feature
        route_feature, mean, var = layers.batch_norm_moving(
            "route_feature_batch_norm",
            self.route_feature,
            rescale=False,
            is_training=self.is_training
        )
        main_diff, mean, var = layers.batch_norm_moving(
            "main_diff_batch_norm",
            self.main_diff,
            rescale=False,
            is_training=self.is_training
        )
        context_feature, mean, var = layers.batch_norm_moving(
            "context_feature_batch_norm",
            self.context_feature,
            rescale=False,
            is_training=self.is_training
        )
        # new: [n,**]
        strategy_feature_part1, mean, var = layers.batch_norm_moving(
            "strategy_feature_batch_norm_part1",
            self.strategy_feature_base[:, :4],  # (dis,odf,of,df)
            rescale=False,
            is_training=self.is_training
        )
        strategy_feature_part3, mean, var = layers.batch_norm_moving(
            "strategy_feature_batch_norm_part3",
            self.strategy_feature_base[:, -15:],  # min/max/avg of (time,dis,lights,toll,highway) in recalls
            rescale=False,
            is_training=self.is_training
        )
        strategy_feature = tf.concat([strategy_feature_part1,
                                      self.strategy_feature_base[:, 4:-15],
                                      strategy_feature_part3], axis=1)

        # ------------------------------ step 3:  concat features  ------------------------------
        feature_list = [
            route_feature,
            user_feature,
            main_diff,
            context_feature
        ]
        final_feature_concat = tf.concat(
            feature_list,
            axis=1
        )
        total_feature_num = int(final_feature_concat.shape[1])

        # -------------------------- step 4:  get expert attention score  --------------------------
        strategy_attention_score = self.strategy_attention("strategy_attention", strategy_feature)
        self.strategy_fc2 = strategy_attention_score

        # ------------------------- step 5:  scenario aware feature filter  -------------------------
        if dsfnet_configer.USE_SAFF:
            final_feature_filter = self.scenario_aware_feature_filter("saff_1", strategy_feature, final_feature_concat,
                                                                      total_feature_num)
        else:
            final_feature_filter = final_feature_concat

        # --------------------------- step 6:  dfsnet base model details  ---------------------------
        self.orth_cost = [0., 0.]
        # layer 1
        hidden1, weights = layers.neuron_layer_partitioned(
            "final_hidden1",
            final_feature_filter,
            self.expert_neuron0,
            activation='relu',
            if_batch_norm=False,
            is_training=self.is_training,
            cur_domain_id=-1,
            n_domain=self.expert_num,
            n_inputs=total_feature_num,
            strategy_feature=None,
            embedding_weight=strategy_attention_score
        )
        temp = self.get_orth_cost(weights)
        self.orth_cost[0] += temp[0]
        self.orth_cost[1] += temp[1]
        if dsfnet_configer.USE_BN:
            hidden1_sn, mean, var = layers.batch_norm(
                "sn_1",
                input_feature=hidden1,
                is_training=self.is_training
            )
        else:
            hidden1_sn = hidden1
        # layer 2
        hidden2, weights = layers.neuron_layer_partitioned(
            "final_hidden2",
            hidden1_sn,
            self.expert_neuron1,
            activation='relu',
            if_batch_norm=False,
            is_training=self.is_training,
            cur_domain_id=-1,
            n_domain=self.expert_num,
            n_inputs=self.expert_neuron0,
            strategy_feature=None,
            embedding_weight=strategy_attention_score
        )
        temp = self.get_orth_cost(weights)
        self.orth_cost[0] += temp[0]
        self.orth_cost[1] += temp[1]
        if dsfnet_configer.USE_BN:
            hidden2_sn, mean, var = layers.batch_norm(
                "sn_2",
                input_feature=hidden2,
                is_training=self.is_training
            )
        else:
            hidden2_sn = hidden2
        # layer 3
        hidden3, weights = layers.neuron_layer_partitioned(
            "final_hidden3",
            hidden2_sn,
            self.expert_neuron2,
            activation='relu',
            if_batch_norm=False,
            is_training=self.is_training,
            cur_domain_id=-1,
            n_domain=self.expert_num,
            n_inputs=self.expert_neuron1,
            strategy_feature=None,
            embedding_weight=strategy_attention_score
        )
        temp = self.get_orth_cost(weights)
        self.orth_cost[0] += temp[0]
        self.orth_cost[1] += temp[1]
        if dsfnet_configer.USE_BN:
            hidden3_sn, mean, var = layers.batch_norm(
                "sn_3",
                input_feature=hidden3,
                is_training=self.is_training
            )
        else:
            hidden3_sn = hidden3
        # layer 4
        hidden4, weights = layers.neuron_layer_partitioned(
            "final_hidden4",
            hidden3_sn,
            32,
            activation='relu',
            if_batch_norm=False,
            is_training=self.is_training,
            cur_domain_id=-1,
            n_domain=self.expert_num,
            n_inputs=self.expert_neuron2,
            strategy_feature=None,
            embedding_weight=strategy_attention_score
        )
        temp = self.get_orth_cost(weights)
        self.orth_cost[0] += temp[0]
        self.orth_cost[1] += temp[1]
        hidden4_sn = hidden4
        if dsfnet_configer.USE_SABN:
            hidden4_sn = self.secnario_aware_batch_normalization("sabn_1", strategy_feature, hidden4_sn)
        # layer 5
        self.task_out, weights = layers.neuron_layer_partitioned(
            "task_out",
            hidden4_sn,
            1,
            cur_domain_id=-1,
            n_domain=self.expert_num,
            n_inputs=32,
            strategy_feature=None,
            embedding_weight=strategy_attention_score
        )
        temp = self.get_orth_cost(weights)
        self.orth_cost[0] += temp[0]
        self.orth_cost[1] += temp[1]

        # ----------------------------- step 6:  cost and predict  -----------------------------
        self.cost = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.task_out,
            labels=self.label
        )
        self.cost = tf.reduce_mean(self.cost)

        if dsfnet_configer.ORTH_MAT:
            self.orth_lambda0 = 0.1
            # self.orth_lambda1=0.01
            self.orth_cost[0] *= self.orth_lambda0
            # self.orth_cost[1]*=self.orth_lambda1
            self.ce_cost = self.cost
            self.cost += self.orth_cost[0]  # +self.orth_cost[1]
            self.cost = {'loss': self.cost,
                         'orthMat_loss': self.orth_cost[0],
                         'ce_loss': self.ce_cost}
        elif dsfnet_configer.ONLY_FCR:
            self.orth_lambda0 = 0.1
            # self.orth_lambda1=0.01
            self.orth_cost[0] *= self.orth_lambda0
            # self.orth_cost[1]*=self.orth_lambda1
            self.ce_cost = self.cost
            self.cost += self.orth_cost[0]  # +self.orth_cost[1]
            self.cost = {'loss': self.cost,
                         'fcr_loss': self.orth_cost[0],
                         'ce_loss': self.ce_cost}
        elif dsfnet_configer.MMA_INSTEAD_FCR:  # MMA+CPC
            self.orth_lambda0 = 0.03  # 0.03
            self.orth_lambda1 = 0.01
            self.orth_cost[0] *= self.orth_lambda0
            self.orth_cost[1] *= self.orth_lambda1
            self.ce_cost = self.cost
            self.cost += self.orth_cost[0] + self.orth_cost[1]
            self.cost = {'loss': self.cost,
                         'mma_loss': self.orth_cost[0],
                         'cpc_loss': self.orth_cost[1],
                         'ce_loss': self.ce_cost}
        elif dsfnet_configer.ANGLE_CENTROID:  # FCR+CPC
            self.orth_lambda0 = 0.01  # 0.03
            self.orth_lambda1 = 0.01
            self.orth_cost[0] *= self.orth_lambda0
            self.orth_cost[1] *= self.orth_lambda1
            self.ce_cost = self.cost
            self.cost += self.orth_cost[0] + self.orth_cost[1]
            self.cost = {'loss': self.cost,
                         'fcr_loss_angleCentroid': self.orth_cost[0],
                         'cpc_loss_angleCentroid': self.orth_cost[1],
                         'ce_loss': self.ce_cost}
        else:  # vanilla FCR+CPC
            self.orth_lambda0 = 0.01  # 0.03
            self.orth_lambda1 = 0.01  # 0.01
            self.orth_cost[0] *= self.orth_lambda0
            self.orth_cost[1] *= self.orth_lambda1
            self.ce_cost = self.cost
            self.cost += self.orth_cost[0] + self.orth_cost[1]
            self.cost = {'loss': self.cost,
                         'fcr_loss': self.orth_cost[0],
                         'cpc_loss': self.orth_cost[1],
                         'ce_loss': self.ce_cost}

        self.learning_rate = tf.train.exponential_decay(
            learning_rate=dsfnet_configer.learning_rate,
            global_step=dsfnet_configer.num_epoch,
            decay_steps=dsfnet_configer.decay_steps,
            decay_rate=dsfnet_configer.decay_rate,
            staircase=False
        )
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.train.AdamOptimizer(
                self.learning_rate
            ).minimize(self.cost['loss'], global_step=dsfnet_configer.num_epoch)
        self.predict = tf.nn.sigmoid(self.task_out, name="out_value")
        self.auc = tf.py_func(
            roc_auc_score, [self.label, self.predict], tf.double)

    def embed_user(self, ):
        '''
        Get user feature embedding
        '''
        with tf.variable_scope("embed_user"):
            j_type = tf.cast(self.user_feature[:, dsfnet_configer.JOB_FEAT_INDEX], dtype=tf.int32)
            job_embedding = tf.get_variable(name='job_embedding',
                                            shape=(dsfnet_configer.JOB_NUM, dsfnet_configer.JOB_EMBED_DIM))
            job_feat = tf.nn.embedding_lookup(job_embedding, j_type)
            self.user_feature = tf.concat([self.user_feature[:, :dsfnet_configer.JOB_FEAT_INDEX],
                                           job_feat], axis=-1)
            print("User embedding done: self.user_feature.shape =", self.user_feature.shape)

    def embed_strategy(self):
        '''
        Get scene feature embedding
        '''
        with tf.variable_scope("embed_strategy"):
            poi_type = tf.cast(self.strategy_feature[:, dsfnet_configer.POI_FEAT_INDEX], dtype=tf.int32)
            poi_embedding = tf.get_variable(name='poi_embedding',
                                            shape=(dsfnet_configer.POI_NUM, dsfnet_configer.POI_EMBED_DIM))
            poi_feat = tf.nn.embedding_lookup(poi_embedding, poi_type)
            day_type = tf.cast(self.strategy_feature[:, dsfnet_configer.DAY_FEAT_INDEX], dtype=tf.int32)
            day_feat = tf.concat([tf.expand_dims(day_type, axis=-1)] * dsfnet_configer.DAY_EMBED_DIM, axis=-1)
            if dsfnet_configer.BASELINE_COMPARISON:
                self.strategy_feature = tf.concat([self.strategy_feature[:, :dsfnet_configer.POI_FEAT_INDEX],
                                                   poi_feat, day_feat], axis=-1)  # (n,14)
            else:
                or_type = tf.cast(self.strategy_feature[:, dsfnet_configer.OR_FEAT_INDEX], dtype=tf.int32)
                or_embedding = tf.get_variable(name='or_embedding',
                                               shape=(dsfnet_configer.ODR_NUM, dsfnet_configer.OR_EMBED_DIM))
                or_feat = tf.nn.embedding_lookup(or_embedding, or_type)
                dr_type = tf.cast(self.strategy_feature[:, dsfnet_configer.DR_FEAT_INDEX], dtype=tf.int32)
                dr_embedding = tf.get_variable(name='dr_embedding',
                                               shape=(dsfnet_configer.ODR_NUM, dsfnet_configer.DR_EMBED_DIM))
                dr_feat = tf.nn.embedding_lookup(dr_embedding, dr_type)
                self.strategy_feature = tf.concat([self.strategy_feature[:, :dsfnet_configer.POI_FEAT_INDEX],
                                                   poi_feat, day_feat, or_feat, dr_feat,
                                                   self.strategy_feature[:, dsfnet_configer.DR_FEAT_INDEX + 1:]],
                                                  axis=-1)  # (n,26)
            print("Strategy embedding done: self.strategy_feature.shape =", self.strategy_feature.shape)

    def secnario_aware_batch_normalization(self, name, strategy_feature, input_feature):
        '''
        Secnario Aware Batch Normalization:
        Try to learn the feature distribution transformation from scenario-agnostic distribution to scenario-specific one
        '''
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            beta_fc1, _ = layers.neuron_layer(
                "beta_fc1",
                strategy_feature,
                self.dynamic_neuron0,
                activation='relu',
                if_batch_norm=True,
                is_training=self.is_training
            )
            beta_fc2, _ = layers.neuron_layer(
                "beta_fc2",
                beta_fc1,
                1
            )
            self.beta_fc2 = beta_fc2
            gamma_fc1, _ = layers.neuron_layer(
                "gamma_fc1",
                strategy_feature,
                self.dynamic_neuron0,
                activation='relu',
                if_batch_norm=True,
                is_training=self.is_training
            )
            gamma_fc2, _ = layers.neuron_layer(
                "gamma_fc2",
                gamma_fc1,
                1
            )
            self.gamma_fc2 = gamma_fc2
            output_feature = tf.multiply(input_feature, self.gamma_fc2) + self.beta_fc2
            return output_feature

    def scenario_aware_feature_filter(self, name, strategy_feature, input_feature, total_feature_num):
        '''
        Scenario Aware Feature Filter:
        Try to model the process of humans manually selecting features based on scenario information
        '''
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            feature_fc1, _ = layers.neuron_layer(
                "feature_fc1",
                tf.concat([input_feature, strategy_feature], axis=1),
                self.dynamic_neuron0,
                activation='relu',
                if_batch_norm=True,
                is_training=self.is_training
            )
            feature_fc2, _ = layers.neuron_layer(
                "feature_fc2",
                feature_fc1,
                total_feature_num
            )
            final_feature_filter = tf.multiply(input_feature, feature_fc2)
            return final_feature_filter

    def strategy_attention(self, name, strategy_feature):
        '''
        Get Scene Attention Score
        '''
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            strategy_fc1, _ = layers.neuron_layer(
                "strategy_fc1",
                strategy_feature,
                self.dynamic_neuron0,
                activation='relu',
                if_batch_norm=True,
                is_training=self.is_training
            )
            strategy_fc2, _ = layers.neuron_layer(
                "strategy_fc2",
                strategy_fc1,
                self.expert_num
            )
            return strategy_fc2

    def orth_mat(self, weights_bar):
        return tf.sqrt(tf.reduce_sum(
            tf.square(tf.eye(self.n_domain) - tf.matmul(weights_bar, tf.transpose(weights_bar, perm=[1, 0])))))

    def expert_mma(self, weights_bar):
        """
        weights_bar: #(n_domain,in)
        """
        PI = 3.141592654
        MAX_CLIP = 0.99999

        theta = tf.acos(tf.clip_by_value(tf.matmul(weights_bar, tf.transpose(weights_bar, perm=[1, 0])),
                                         -MAX_CLIP, MAX_CLIP))  # (n_domain,n_domain)
        return -tf.reduce_mean(tf.reduce_min(theta + 2 * PI * tf.eye(self.n_domain), axis=-1))

    def contrastive_prototype_clustering(self, weights_bar, weights_tot, out_dim):
        """
        weights_bar: #(n_domain,in)
        weights_tot: #(n_domain*out,in)
        """
        PI = 3.141592654
        L = 20
        MAX_CLIP = 0.99999
        DELTA = tf.acos(1 / (1 - self.n_domain)) / L

        theta = tf.acos(tf.clip_by_value(tf.matmul(weights_tot, tf.transpose(weights_bar, perm=[1, 0])),
                                         -MAX_CLIP, MAX_CLIP))  # (n_domain*out,n_domain)
        mask = np.zeros((self.n_domain * out_dim, self.n_domain), dtype=np.float32)
        mask[range(self.n_domain * out_dim), [i for _ in range(out_dim) for i in range(self.n_domain)]] = 1.
        mask = tf.constant(mask)
        theta_pos = tf.reduce_max(mask * theta, axis=0)  # (n_domain)
        theta = tf.gather(theta, indices=tf.argmax(mask * theta, axis=0), axis=0)  # (n_domain,n_domain)
        theta_neg = tf.reduce_min(2 * PI * tf.eye(self.n_domain) + theta, axis=1)  # (n_domain)
        return tf.reduce_mean(tf.maximum(0., DELTA + theta_pos - theta_neg))

    def cosine_contrastive_prototype_clustering(self, weights_bar, weights_tot, out_dim):
        """
        weights_bar: #(n_domain,in)
        weights_tot: #(n_domain*out,in)
        """
        PI = 3.141592654
        DELTA = 0.8  # 0.2

        theta = tf.matmul(weights_tot, tf.transpose(weights_bar, perm=[1, 0]))  # (n_domain*out,n_domain)
        mask = np.zeros((self.n_domain * out_dim, self.n_domain), dtype=np.float32)
        mask[range(self.n_domain * out_dim), [i for _ in range(out_dim) for i in range(self.n_domain)]] = 1.
        mask = tf.constant(mask)
        theta_pos = tf.reduce_min(mask * theta, axis=0)  # (n_domain)
        theta = tf.gather(theta, indices=tf.argmin(mask * theta, axis=0), axis=0)  # (n_domain,n_domain)
        theta_neg = tf.reduce_max(2 * PI * tf.eye(self.n_domain) + theta, axis=1)  # (n_domain)
        return tf.reduce_mean(tf.minimum(0., DELTA - theta_pos + theta_neg))

    def factor_centroid_repulsion(self, weights_bar):
        """
        weights_bar: #(n_domain,in)
        """
        PI = 3.141592654
        MAX_CLIP = 0.99999

        theta = tf.acos(tf.clip_by_value(tf.matmul(weights_bar, tf.transpose(weights_bar, perm=[1, 0])),
                                         -MAX_CLIP, MAX_CLIP))  # (n_domain,n_domain)
        return tf.reduce_mean(
            tf.reduce_max((1. - tf.eye(self.n_domain)) * tf.square(theta - tf.acos(1. / (1 - self.n_domain))), axis=-1))

    def get_orth_cost(self, weights):
        """
        FCR + CPC
        note: #in >= n_domain-1
        return: [car, triplet]
        """
        EPS = 1e-12
        costs = []
        weights = weights[1:self.n_domain + 1]  # (n_domain,in,out), in>out
        if dsfnet_configer.ANGLE_CENTROID:
            weights_normed = weights / tf.expand_dims(EPS + tf.sqrt(
                tf.reduce_sum(tf.square(weights), axis=1)), axis=1)  # normed
            weights_bar = tf.reduce_mean(weights_normed, axis=-1)  # (n_domain,in)
            weights_bar = weights_bar / tf.expand_dims(EPS + tf.sqrt(
                tf.reduce_sum(tf.square(weights_bar), axis=-1)), axis=-1)  # normed again
        else:
            weights_bar = tf.reduce_mean(weights, axis=-1)  # (n_domain,in)
            weights_bar = weights_bar / tf.expand_dims(EPS + tf.sqrt(
                tf.reduce_sum(tf.square(weights_bar), axis=-1)), axis=-1)  # normed
        if dsfnet_configer.ORTH_MAT:
            costs += [self.orth_mat(weights_bar)]
        elif dsfnet_configer.MMA_INSTEAD_FCR:
            costs += [self.expert_mma(weights_bar)]
        else:  # FCR
            costs += [self.factor_centroid_repulsion(weights_bar)]
        out_dim = int(weights.shape[2])
        weights_tot = tf.reshape(tf.transpose(weights, perm=[0, 2, 1]),
                                 [self.n_domain * out_dim, -1])  # (n_domain*out,in)
        weights_tot = weights_tot / tf.expand_dims(EPS + tf.sqrt(
            tf.reduce_sum(tf.square(weights_tot), axis=-1)), axis=-1)  # normed
        if dsfnet_configer.COSINE_CNC:
            costs += [self.cosine_contrastive_prototype_clustering(weights_bar, weights_tot, out_dim)]
        else:
            costs += [self.contrastive_prototype_clustering(weights_bar, weights_tot, out_dim)]
        return costs
