# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import math


def get_feat_size(Params, dataloaders, features):
    kernel = Params["kernel"]
    stride = Params["stride"]
    model_name = Params["Model_name"]
    pooling_layer = Params["pooling_layer"]
    agg_func = Params["agg_func"]
    feature_height = 7


    if agg_func == "local":
        pooling_output = math.floor((feature_height - kernel) / stride) + 1
        num_ftrs = features * pooling_output * pooling_output
    elif agg_func == "global":
        num_ftrs = features
    

    return num_ftrs