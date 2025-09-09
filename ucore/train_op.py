# ===========================
# -*- coding: utf-8 -*-
# Author: DSFNet Authors
# Date: 2024-09-27
# Keep Coding, Keep Thinking
# ===========================


import sys
import time
import logging
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from corona_train.abstract.abstract_train_op import abstract_train_op

from numpy.random import seed
from tensorflow import set_random_seed

from .train.parser import Parser
from .config import dsfnet_configer
from .train.data_reader import DataReaderExpertV1

seed(1)
set_random_seed(1)

logging.basicConfig(
    format="[ %(asctime)s  %(levelname)s] %(message)s",
    datefmt="%a %Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class FileWriter:
    def __init__(self, out_path, graph):
        if dsfnet_configer.DATA_SET_OSS:
            self.summary_writer = tf.summary.MetricsWriter(out_path, graph)
        else:
            self.summary_writer = tf.summary.FileWriter(out_path, graph)

    def add_info(self, info, step, name=""):
        if dsfnet_configer.DATA_SET_OSS:
            self.summary_writer.add_scalar(tag=name, value=info, step=step)
        else:
            summary = tf.Summary()
            summary.value.add(tag=name, simple_value=info)
            self.summary_writer.add_summary(summary, step)


def cal_target(summary_writer, data_res, step):
    '''
    Compute True Positive, False Positive, True Negative and False Negative
    '''
    res_threshold_list = []
    res_negative_recall_list = []
    record_90 = False
    record_85 = False
    record_80 = False
    threshold_list = [i * 0.001 for i in range(1, 1001)]
    for index, threshold in enumerate(threshold_list):
        threshold = round(threshold, 3)

        TP = data_res["score_list"][
            (data_res["score_list"] >= threshold) & (data_res['label_list'] == 1)].count()  # True Positive
        FP = data_res["score_list"][
            (data_res["score_list"] >= threshold) & (data_res['label_list'] == 0)].count()  # False Positive
        TN = data_res["score_list"][
            (data_res["score_list"] < threshold) & (data_res['label_list'] == 0)].count()  # True Negative
        FN = data_res["score_list"][
            (data_res["score_list"] < threshold) & (data_res['label_list'] == 1)].count()  # False Negative

        total_positive = TP + FN
        predict_positive = TP + FP
        total_negative = TN + FP
        predict_negative = TN + FN

        positive_recall = TP / total_positive
        positive_precision = TP / predict_positive if predict_positive > 0 else 0
        negative_recall = TN / total_negative
        negative_precision = TN / predict_negative if predict_negative > 0 else 0
        accuracy = (TP + TN) / (TP + FP + TN + FN)

        summary_writer.add_info(positive_recall, int(threshold * 1000),
                                "reacll_precision_{}/positive_recall".format(str(step)))
        summary_writer.add_info(positive_precision, int(threshold * 1000),
                                "reacll_precision_{}/positive_precision".format(str(step)))
        summary_writer.add_info(negative_recall, int(threshold * 1000),
                                "reacll_precision_{}/negative_recall".format(str(step)))
        summary_writer.add_info(negative_precision, int(threshold * 1000),
                                "reacll_precision_{}/negative_precision".format(str(step)))
        summary_writer.add_info(accuracy, int(threshold * 1000), "reacll_precision_{}/accuracy".format(str(step)))
        # print("{:.3f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(threshold, positive_recall, positive_precision, negative_recall, negative_precision))
        if negative_precision < 0.9 and not record_90:
            res_threshold_list.append(threshold * 100)
            res_negative_recall_list.append(negative_recall)
            record_90 = True
        if negative_precision < 0.85 and not record_85:
            res_threshold_list.append(threshold * 100)
            res_negative_recall_list.append(negative_recall)
            record_85 = True
        if negative_precision < 0.8 and not record_80:
            res_threshold_list.append(threshold * 100)
            res_negative_recall_list.append(negative_recall)
            record_80 = True

        if index % 100 == 0:
            logger.info("cal target step: {}".format(index))

    return res_threshold_list, res_negative_recall_list


class BatchEvaluatorBase(object):
    def __init__(self, data_reader, sess, nn_model):
        self.data_reader = data_reader
        self.sess = sess
        self.nn_model = nn_model

    def get_eval(self):
        cost_avg, auc_avg, cost_avg_dict, auc_avg_list, data_res_pd = self.get_eval_metrics_batch()
        return cost_avg, auc_avg, cost_avg_dict, auc_avg_list, data_res_pd

    def get_eval_metrics_batch(self):
        cost_list = []
        predict_list = []
        model_score_list = []
        label_list = []

        self.sess.run(self.data_reader.eval_iterator.initializer)
        next_element = self.data_reader.eval_iterator.get_next()
        i_step = 1
        try:
            while True:
                if i_step % 1000 == 0:
                    logger.info("eval step: {}".format(i_step))

                cost, task_out, model_score, label = self._sess_run_batch(next_element)
                cost_list.append(cost if not isinstance(cost, dict) else cost['loss'])
                model_score_list.extend(model_score)
                predict_list.extend(task_out)
                label_list.extend(label)
                i_step += 1
        except tf.errors.OutOfRangeError:
            logger.info('eval complete')

        if len(cost_list) > 0:
            cost_avg = sum(cost_list) / len(cost_list)
            auc_avg = roc_auc_score(label_list, model_score_list)
        else:
            cost_avg = 0
            auc_avg = 0

        data_res_pd = pd.DataFrame({'score_list': model_score_list[:500000], "label_list": label_list[:500000]})

        return cost_avg, auc_avg, {}, {}, data_res_pd

    def _sess_run_batch(self, next_element):
        raise NotImplementedError


class BatchEvaluatorExpert(BatchEvaluatorBase):
    def __init__(self, data_reader, sess, nn_model):
        super(BatchEvaluatorExpert, self).__init__(data_reader, sess, nn_model)

    def _sess_run_batch(self, next_element):
        eval_route_feature, \
        eval_main_diff, \
        eval_context_feature, \
        eval_strategy_feature, \
        eval_label, \
        eval_coverage = self.sess.run(next_element)
        context_feature_mask_index = [28, 29]
        eval_context_feature[:, context_feature_mask_index] = 0
        cost, task_out, model_score, label = self.sess.run(
            [
                self.nn_model.cost,
                self.nn_model.task_out,
                self.nn_model.predict,
                self.nn_model.label
            ],
            feed_dict={
                self.nn_model.route_feature: eval_route_feature,
                self.nn_model.main_diff: eval_main_diff,
                self.nn_model.context_feature: eval_context_feature,
                self.nn_model.strategy_feature: eval_strategy_feature,
                self.nn_model.label: eval_label,
                self.nn_model.is_training: False,
                self.nn_model.coverage: eval_coverage,
            }
        )

        return cost, task_out, model_score, label


class TrainBase(object):
    def __init__(self, data_reader, batch_class_name):
        self.data_reader = data_reader
        logger.info("read data finished")
        self.nn_model = eval(dsfnet_configer.model_class_name)(dsfnet_configer)
        self.batch_class_name = batch_class_name
        self.log_info = True
        self.out_global_step = None
        self.data_time = 0
        self.train_time = 0

    def run_train(self):
        init = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            logger.info("init model finished: {}".format(dsfnet_configer.model_class_name))
            sess.run(tf.global_variables_initializer())
            logger.info("init iterator finished")
            sess.run(tf.tables_initializer())
            logger.info("init tabel finished")
            coord = tf.train.Coordinator()
            runners = tf.train.start_queue_runners(coord=coord)
            try:
                step = 0
                summary_writer = FileWriter(dsfnet_configer.tensorboard_path, sess.graph)
                model_saver = tf.train.Saver(max_to_keep=100)
                batch_evaluator = eval(self.batch_class_name)(self.data_reader, sess, self.nn_model)
                logger.info("to start while loop")

                sess.run(self.data_reader.train_iterator.initializer)
                next_element = self.data_reader.train_iterator.get_next()
                while not coord.should_stop():
                    step += 1
                    if step % dsfnet_configer.STEP_PRINT_HEART == 0:
                        logger.info("+++++++++++++++++++++++++++++++")
                        logger.info("step: {}".format(step))
                        logger.info("data_time: {}, train_time: {}".format(self.data_time, self.train_time))
                        self.data_time = 0
                        self.train_time = 0
                    self._train_step(sess, next_element, self.nn_model)
                    if step == 10 or step % dsfnet_configer.STEP_PRINT_EVAL == 0:
                        scene_logits, train_cost, train_learning_rate, train_auc, eval_loss, eval_auc, eval_loss_dict, eval_auc_dict, data_res_pd = self._eval_step(
                            sess, self.nn_model, batch_evaluator)
                        logger.info(f"The logit of each experts in 1'st sample: {scene_logits[0]}")
                        logger.info(
                            f"The sigmoid gating of each experts in 1'st sample: {tf.sigmoid(scene_logits[0]).eval(session=tf.Session())}")
                        logger.info(
                            f"The softmax gating of each experts in 1'st sample: {tf.nn.softmax(scene_logits[0], axis=0).eval(session=tf.Session())}")
                        if isinstance(train_cost, dict):
                            for cost_name, cost_value in train_cost.items():
                                logger.info("train_{}: {}, {}".format(cost_name, cost_value, train_auc))
                                summary_writer.add_info(cost_value, step, f"train/train_{cost_name}")
                        else:
                            logger.info("train: {}, {}".format(train_cost, train_auc))
                            summary_writer.add_info(train_cost, step, f"train/train_loss")
                        logger.info("eval: {}, {}".format(eval_loss, eval_auc))
                        summary_writer.add_info(train_auc, step, "train/train_auc")
                        summary_writer.add_info(train_learning_rate, step, "train/learning_rate")
                        summary_writer.add_info(eval_loss, step, "test/eval_loss")
                        summary_writer.add_info(eval_auc, step, "test/eval_auc")

                        if step == 10 or step % dsfnet_configer.STEP_CAL_RECALL == 0:
                            res_threshold_list, res_negative_recall_list = cal_target(summary_writer, data_res_pd, step)
                            if len(res_threshold_list) == 3 and len(res_negative_recall_list) == 3:
                                summary_writer.add_info(res_threshold_list[0], step, "filter_threshold/90")
                                summary_writer.add_info(res_threshold_list[1], step, "filter_threshold/85")
                                summary_writer.add_info(res_threshold_list[2], step, "filter_threshold/80")

                                summary_writer.add_info(res_negative_recall_list[0], step, "filter_recall/90")
                                summary_writer.add_info(res_negative_recall_list[1], step, "filter_recall/85")
                                summary_writer.add_info(res_negative_recall_list[2], step, "filter_recall/80")

                        model_saver.save(sess, '{}/step_{}'.format(dsfnet_configer.model_path, step), global_step=step)
            except tf.errors.OutOfRangeError:
                logger.info('epochs complete')
            coord.request_stop()
            coord.join(runners)
            logger.info('train op end')

    def _train_step(self, sess, next_element, nn_model):
        raise NotImplementedError

    def _eval_step(self, sess, nn_model, batch_evaluator):
        raise NotImplementedError


class TrainExpert(TrainBase):
    def __init__(self, data_reader, batch_class_name):
        super(TrainExpert, self).__init__(data_reader, batch_class_name)

    def _train_step(self, sess, next_element, nn_model, global_step=None):
        '''
        Train DSFNet for one step
        '''
        time_start = time.time()
        self.train_route_feature, \
        self.train_main_diff, \
        self.train_context_feature, \
        self.train_strategy_feature, \
        self.train_label, \
        self.train_coverage = sess.run(next_element)
        time_data_finished = time.time()
        self.data_time += time_data_finished - time_start

        context_feature_mask_index = [28, 29]
        self.train_context_feature[:, context_feature_mask_index] = 0
        if self.log_info:
            logger.info("route_shape of data: {}".format(str(self.train_route_feature.shape)))
            logger.info("main_diff of data: {}".format(str(self.train_main_diff.shape)))
            logger.info("context_shape of data: {}".format(str(self.train_context_feature.shape)))
            logger.info("strategy_shape of data: {}".format(str(self.train_strategy_feature.shape)))
        self.log_info = False

        sess.run(
            [nn_model.optimizer],
            feed_dict={
                nn_model.route_feature: self.train_route_feature,
                nn_model.main_diff: self.train_main_diff,
                nn_model.context_feature: self.train_context_feature,
                nn_model.strategy_feature: self.train_strategy_feature,
                nn_model.label: self.train_label,
                nn_model.is_training: True,
                nn_model.coverage: self.train_coverage,
            }
        )
        time_train_finished = time.time()
        self.train_time += time_train_finished - time_data_finished

    def _eval_step(self, sess, nn_model, batch_evaluator):
        train_cost, train_learning_rate, train_auc, scene_logits = sess.run(
            [nn_model.cost, nn_model.learning_rate, nn_model.auc, nn_model.strategy_fc2],
            feed_dict={
                nn_model.route_feature: self.train_route_feature,
                nn_model.main_diff: self.train_main_diff,
                nn_model.context_feature: self.train_context_feature,
                nn_model.strategy_feature: self.train_strategy_feature,
                nn_model.label: self.train_label,
                nn_model.is_training: False,
                nn_model.coverage: self.train_coverage,
            }
        )
        eval_loss, eval_auc, eval_loss_dict, eval_auc_dict, data_res_pd = batch_evaluator.get_eval()
        return scene_logits, train_cost, train_learning_rate, train_auc, eval_loss, eval_auc, eval_loss_dict, eval_auc_dict, data_res_pd

class train_op(abstract_train_op):
    def train(self):
        Parser().parse()
        logger.info('In train_op, model name: {}'.format(dsfnet_configer.model_class_name))
        data_reader = DataReaderExpertV1(self)
        batch_class_name = "BatchEvaluatorExpert"
        trainer = TrainExpert(data_reader, batch_class_name)
        trainer.run_train()
        self.write_file_path("{}/{}".format(dsfnet_configer.model_class_name, "step_10-10"))
