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
import sys
from  numpy.random import randint
from config import *
from util import *

from  model import *

#get()

if __name__=='__main__':
    val_x,val_label = getRealVal()
    val_history = []
    val_acc_max = 0;val_best_pos=-1
    saver = tf.train.Saver()
    op = tf.train.AdamOptimizer(0.00001).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,modelfile)
    for x,lb in getRealData():
        ls,_,ac = sess.run([loss,op,acc], feed_dict={input:np.float32(x),label:lb})
        print(ls,ac)
        if int(time.time()%50)==0:
            val_loss, val_acc = sess.run([loss,acc], feed_dict={input: np.float32(val_x), label: val_label})
            val_history+=[val_acc]
            if val_acc>val_acc_max:
                print("best val acc:"+str(val_acc))
                val_acc_max = val_acc
                val_best_pos = len(val_history)-1
                saver.save(sess, modelfile)
            elif len(val_history) - val_best_pos >30:
                sys.exit()

#aa = Image.open(fp).convert('RGB')