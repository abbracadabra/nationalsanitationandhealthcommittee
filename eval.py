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

x,lb = getRealVal()

saver = tf.train.import_meta_graph(modelfile+'.meta')
sess = tf.Session()
#sess.run(tf.global_variables_initializer())
saver.restore(sess,modelfile)
input = tf.get_default_graph().get_tensor_by_name('input:0')
label = tf.get_default_graph().get_tensor_by_name('label:0')
#predix = tf.get_default_graph().get_tensor_by_name('pred:0')
acc = tf.get_default_graph().get_tensor_by_name('acc:0')
_acc = sess.run([acc],feed_dict={input:np.float32(x),label:lb})
print(_acc)