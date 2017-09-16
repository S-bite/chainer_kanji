##coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pickle
from glob import glob
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from PIL import Image

from tqdm import tqdm

class MyGLView(gl.GLViewWidget):
    def paintGL(self,*args, **kwds):
        gl.GLViewWidget.paintGL(self, *args, **kwds)
        self.qglColor(QtCore.Qt.white)
        for pos,char in zip(self.pos,self.char):
            #print(int(pos[0]))
            x=float(pos[0])
            y=float(pos[1])
            z=float(pos[2])
            self.renderText(x,y,z,char)

chars=None
with open("features.pickle","rb") as f:
    chars=pickle.load(f)
d=[]
images=[]
for ch,char in tqdm(chars.items()):
    d.append(char.data[0])


pca=PCA(3)
transed=pca.fit_transform(d)

app = QtGui.QApplication([])
w = MyGLView()
w.pos=transed/2000
w.char=chars
#w.opts['distance'] = 20
g = gl.GLGridItem()
g.scale(10, 10, 10)
w.addItem(g)
sp1 = gl.GLScatterPlotItem(pos=transed/2000, size=16, )
w.addItem(sp1)
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
w.paintGL()
w.show()
QtGui.QApplication.instance().exec_()
