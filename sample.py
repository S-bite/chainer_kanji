# coding:utf-8
from tqdm import tqdm
import os

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.functions.loss.mean_squared_error import mean_squared_error

from PIL import Image
import glob


distance_min = [10**10, None, None]
distance_max = [0, None, None]

for a in "人":  # tqdm(chars[0]):
    for b in chars:
        if a == b:
            continue
        x = model.feature(a)
        y = model.feature(b)
        distance = model.distance(x, y)
        if distance > distance_max[0]:
            distance_max = [distance, a, b]
        if distance < distance_min[0]:
            distance_min = [distance, a, b]

print(distance_max)
print(distance_min)
