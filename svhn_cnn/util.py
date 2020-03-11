import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
import random
import os
import cv2
import imageio
import tensorflow as tf
import time
import uuid
from os import path
import traceback
from config import *
from  numpy.random import randint

def getRealData():
    file_list = os.listdir(trainset)
    x = []
    lb = []
    while True:
        try:
            if len(x)>=batchsize:
                x = []
                lb = []
            fn = random.choice(file_list)
            ccs = fn.split('.')[0].split('_')[1].upper()
            fp = os.path.join(trainset, fn)
            im = Image.open(fp).convert('L')
            im = np.array(im)
            im = np.expand_dims(im, axis=-1)
            llbb = [[vb.index(ccs[i]) for i in range(4)]]
            lb += llbb
            x.append(im)
            if len(x) == batchsize:
                yield x, lb
                x = []
                lb = []
        except:
            traceback.print_exc()

def getRealVal():
    file_list = os.listdir(valset)
    x = []
    lb = []
    for fn in file_list:
        try:
            ccs = fn.split('.')[0].split('_')[1].upper()
            fp = os.path.join(valset, fn)
            im = Image.open(fp).convert('L')
            im = np.array(im)
            im = np.expand_dims(im, axis=-1)
            llbb = [[vb.index(ccs[i]) for i in range(4)]]
            lb += llbb
            x.append(im)
        except:
            traceback.print_exc()
    return x,lb


def processim(fp):
    gif = imageio.get_reader(fp)
    im = np.reshape(gif.get_data(0), (40, 135, 4))[:, :, 0:3]
    h, w, c = im.shape
    for i in range(h):
        for j in range(67):
            px = im[i][j]
            if (np.all(px == [153, 128, 204]) or np.all(px == [51, 0, 51]) or np.all(
                    px == [102, 85, 102]) or np.all(
                px == [102, 43, 204]) or np.all(px == [51, 0, 0]) or np.all(px == [51, 0, 204]) or np.all(
                px == [0, 43, 204])
                    or np.all(px == [0, 0, 204]) or np.all(px == [0, 0, 255]) or np.all(px == [0, 0, 0]) or np.all(
                        px == [0, 85, 204]) or np.all(px == [51, 0, 153]) or np.all(px == [0, 0, 51]) or np.all(
                        px == [0, 0, 153])
                    or np.all(px == [0, 0, 102]) or np.all(px == [51, 0, 102]) or np.all(
                        px == [51, 43, 204]) or np.all(
                        px == [51, 43, 102]) or np.all(px == [51, 43, 153]) or np.all(px == [51, 0, 255]) or np.all(
                        px == [51, 43, 51])):
                im[i][j] = [0, 0, 0]
                pass
            else:
                im[i][j] = [255, 255, 255]
        for j in range(67, 135):
            px = im[i][j]
            if (np.all(px == [153, 43, 102]) or np.all(px == [102, 43, 51]) or np.all(
                    px == [102, 43, 102]) or np.all(px == [51, 0, 102]) or np.all(px == [51, 0, 0]) or np.all(
                px == [102, 0, 102]) or np.all(px == [0, 0, 51]) or
                    np.all(px == [102, 43, 153]) or np.all(px == [51, 0, 51]) or np.all(
                        px == [51, 43, 153]) or np.all(px == [51, 0, 153]) or np.all(px == [102, 0, 51]) or np.all(
                        px == [0, 0, 102]) or np.all(px == [102, 0, 153]) or
                    np.all(px == [102, 0, 0]) or np.all(px == [0, 0, 0]) or np.all(px == [51, 43, 102]) or np.all(
                        px == [0, 0, 153])):
                im[i][j] = [0, 0, 0]
                pass
            else:
                im[i][j] = [255, 255, 255]
    im[:, 0:16, :] = [255, 255, 255]
    im[:, 119:, :] = [255, 255, 255]
    im[0:1, :, :] = [255, 255, 255]
    im[35:40, :, :] = [255, 255, 255]
    return im
