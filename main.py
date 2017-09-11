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


class MyChain(Chain):
    def __init__(self):
        h=4096
        super(MyChain, self).__init__(

            el1=L.Linear(64*64, 192),
            dl1=L.Linear(192,64*64),
        )

    def __call__(self, x):
        return self.decode(self.encode(x))

    def encode(self,x):

        h1=F.relu(self.el1(x))

        return h1
    def decode(self,x):
        h1=F.sigmoid(self.dl1(x)/4096)*255
        return h1

def get_kanji_data():
    res = []
    image_dirs = glob.glob("image/*")
    for image_dir in image_dirs:
        img = Image.open(image_dir)
        arrayImg = np.asarray(img).transpose(0, 1).astype(np.float32).reshape(4096)
        res.append(arrayImg)
    return res


def lossfun(x, t):
    return F.mean_absolute_error(x, t)


def val(n):
    return Variable(np.array(n, dtype=np.float32))


x_data = get_kanji_data()[:]
model = MyChain()
model.compute_accuracy = False

optimizer = optimizers.Adam()
optimizer.setup(model)
batch_data = val(np.random.permutation(x_data))
count = 0
while lossfun(model(batch_data), batch_data).data >= 1:
    #optimizer.zero_grads()
    batch_data = val(np.random.permutation(x_data))

    for i in range(10):
        optimizer.update(lossfun, model(batch_data), batch_data)
    count += 1
    print(count, lossfun(model(batch_data), batch_data).data)

    if count % 1 == 0:
        image_dirs = glob.glob("image/*")
        for i in range(10):
            image_dir=image_dirs[i]
            image_name=os.path.basename(image_dir)[:-4]
            image=np.asarray(Image.open(image_dir)).transpose(0, 1).astype(np.float32).reshape(4096)
            image=val([image])
            d = model(image).data.clip(0,255).reshape(64, 64)
            img = Image.fromarray(d).convert('RGB')
            img.save("sample/%05d_%02d_%s.jpg" % (count,i,image_name))
    if count % 10 == 0:

        serializers.save_npz('network/%05d_model.model' % count, model)
        serializers.save_npz('network/%05d_optimizer.state' % count, optimizer)

serializers.save_npz('network/%05d_model.model' % count, model)
serializers.save_npz('network/%05d_optimizer.state' % count, optimizer)
