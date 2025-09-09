# ===========================
# -*- coding: utf-8 -*-
# Author: DSFNet Authors
# Date: 2024-09-27
# Keep Coding, Keep Thinking
# ===========================

# model config
USE_BN = True
USE_SABN = True
USE_SAFF = True

ORTH_MAT = False
MMA_INSTEAD_FCR = False
ONLY_FCR = False
ANGLE_CENTROID = False
COSINE_CNC = False
LOCAL_TRAIN = False

DSFNET_EXPERT_NUM = 7
USE_USER_EMBEDDING = True
EXPERT_NEURONS = [256, 128, 64]
DYNAMIC_NEURONS = [64, 32]

# data config
STEP_PRINT_HEART = 500
STEP_PRINT_EVAL = 15000
STEP_CAL_RECALL = 120000
TABLE_PARALLEL_SIZE = 0
DECODE_PARALLEL_SIZE = 30
DATA_SET_OSS = True

BASELINE_COMPARISON = False
NEW_SFEAT = True
MANUAL_SCENARIO_NUM = 72
POI_FEAT_INDEX = 6  # on splitted strategy feature
POI_EMBED_DIM = 4  # learnable
POI_NUM = 24  # including unk
DAY_FEAT_INDEX = 7
DAY_EMBED_DIM = 4  # mul one-vector
DAY_NUM = 2
if BASELINE_COMPARISON:
    raise NotImplementedError
else:
    SCENARIO_FEAT_FUM = 30 if NEW_SFEAT else 15  # 30 #for DSFNet
    OR_FEAT_INDEX = 8
    OR_EMBED_DIM = 4  # learnable
    DR_FEAT_INDEX = 9
    DR_EMBED_DIM = 4  # learnable
    ODR_NUM = 3

if USE_USER_EMBEDDING:
    SEX_FEAT_INDEX = 0  # on splitted user feature
    SEX_EMBED_DIM = 2  # learnable
    SEX_NUM = 3  # for DSFNet
    AGE_FEAT_INDEX = 1  # on splitted user feature
    AGE_EMBED_DIM = 2  # learnable
    AGE_NUM = 101  # for DSFNet, unk + 1~100
    JOB_FEAT_INDEX = 2  # on splitted user feature
    JOB_EMBED_DIM = 2  # learnable
    JOB_NUM = 13
    USER_FEAT_NUM = 3  # for DSFNet

# train config
decay_steps = 10000
decay_rate = 0.98
read_epochs = 40
shuffle_size = 30
batch_size = 512
learning_rate = 0.001
