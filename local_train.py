# ===========================
# -*- coding: utf-8 -*-
# Author: DSFNet Authors
# Date: 2024-09-27
# Keep Coding, Keep Thinking
# ===========================

import numpy as np

np.set_printoptions(threshold=np.inf)
import pandas as pd

pd.set_option('display.max_columns', None)
from corona_train.core.context import GraphContext

from ucore.config import dsfnet_configer
from ucore.config.column_config import train_columns

dsfnet_configer.read_epochs = 1
dsfnet_configer.batch_size = 52
dsfnet_configer.STEP_PRINT_HEART = 50
dsfnet_configer.STEP_PRINT_EVAL = 50
dsfnet_configer.DATA_SET_OSS = False
dsfnet_configer.LOCAL_TRAIN = True

graph = GraphContext()

print("\n--------------- read data ----------------")

train_set = graph.source_op(table_name="dsfnet_paper_train_data_www_2025", columns=train_columns,
                            limit_num=500)
valid_set = graph.source_op(table_name="dsfnet_paper_test_data_www_2025", columns=train_columns,
                            limit_num=500)

print("\n--------------- transform ----------------")
train_set_ori, valid_set_ori = graph.table_transform_op(train_set, valid_set)

print("\n--------------- unpack ----------------")
train_set, valid_set = graph.unpack_op(train_set_ori, valid_set_ori)

print("\n--------------- start training ----------------")
_, ckp_path = graph.train_op(train_set, valid_set)

graph.save()
