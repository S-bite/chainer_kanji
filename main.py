# coding:utf-8
from tqdm import tqdm

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
        h=128*2
        super(MyChain, self).__init__(
            l1=L.Linear(4096, h),
            l2=L.Linear(h, 4096),
             #l3=L.Linear(256, 1024),
             #l4=L.Linear(1024, 4096),
        )

    def __call__(self, x, feature=False):
        h1 = F.sigmoid(self.l1(x))
        if feature:
            return h1
        h2 = F.sigmoid(self.l2(h1))*255
        return h2


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


x_data = get_kanji_data()[:100]
model = MyChain()
model.compute_accuracy = False

optimizer = optimizers.Adam(0.1)
optimizer.setup(model)
batch_data = val(np.random.permutation(x_data))
count = 0
while lossfun(model(batch_data), batch_data).data >= 0.7:
    #optimizer.zero_grads()
    batch_data = val(np.random.permutation(x_data))

    for i in range(10):
        optimizer.update(lossfun, model(batch_data), batch_data)
    count += 1
    print(count, lossfun(model(batch_data), batch_data).data)

    if count % 100 == 0:
        d = model(val([x_data[1].data]).data).data.clip(0,255).reshape(64, 64)
        img = Image.fromarray(d).convert('RGB')
        img.save("sample/%05d.jpg" % count)
        serializers.save_npz('network/%05d_model.model' % count, model)
        serializers.save_npz('network/%05d_optimizer.state' % count, optimizer)
d = model(val([x_data[43].data]).data).data.reshape(64, 64)
img = Image.fromarray(d).convert('RGB')
img.save("sample/%05d.jpg" % count)
serializers.save_npz('network/%05d_model.model' % count, model)
serializers.save_npz('network/%05d_optimizer.state' % count, optimizer)
