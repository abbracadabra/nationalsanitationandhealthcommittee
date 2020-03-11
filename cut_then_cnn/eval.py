import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
import random
import os
import cv2
import tensorflow as tf
import time
import uuid
from os import path
from  numpy.random import randint
from config import *
from  util import *
#from  model import *

im = processim(r'D:\Users\xxx\Desktop\下载.gif')
x = np.array([im[:,20:44,[0]],im[:,44:68,[0]],im[:,68:92,[0]],im[:,92:116,[0]]])

saver = tf.train.import_meta_graph(modelfile+'.meta')
sess = tf.Session()
#sess.run(tf.global_variables_initializer())
saver.restore(sess,modelfile)
input = tf.get_default_graph().get_tensor_by_name('input:0')
label = tf.get_default_graph().get_tensor_by_name('label:0')
predix = tf.get_default_graph().get_tensor_by_name('pred:0')
acc = tf.get_default_graph().get_tensor_by_name('acc:0')
predices = sess.run([predix],feed_dict={input:x})
print(''.join([vb[i] for i in predices[0]]))
