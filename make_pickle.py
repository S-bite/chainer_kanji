#coding:utf-8
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

import pickle
import json
class MyChain(Chain):
    def __init__(self):
        h = 4096
        super(MyChain, self).__init__(
            el1=L.Linear(64 * 64, 192),
            dl1=L.Linear(192, 64 * 64),


        )

    def __call__(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):

        h1 = F.relu(self.el1(x))
        return h1

    def decode(self, x):
        h1 = F.sigmoid(self.dl1(x) / 4096) * 255
        return h1

    def feature(self, char):
        img = None
        try:
            img = Image.open("image/%s.jpg" % char)
        except Exception:
            raise("invald path")
        img = np.asarray(img, dtype=np.float32).transpose(0, 1).reshape(4096)

        return self.encode(Variable(np.array([img])))

    def save_image(self, x, name):
        d = self.decode(x).data.clip(0, 255).reshape(64, 64)
        img = Image.fromarray(d).convert('RGB')
        img.save(name)

    def distance(self, x, y):
        res = 0
        for a, b in zip(x.data[0], y.data[0]):
            res += (a - b)**2
        return res**(1 / 2)

model = MyChain()
generation = 330
serializers.load_npz('network/%05d_model.model' % generation, model)

model.save_image(model.feature("test"),"res.jpg")
