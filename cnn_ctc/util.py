from config import *
import os
from PIL import Image
import imageio
import numpy as np
import uuid
from os import path
import random
import traceback
import cv2
import tensorflow as tf

def getval():
    """
    :return: val_x, val_label, val_strs
    """
    ixarr = []
    imarr = []
    valarr = []
    strarr = []
    file_list = os.listdir(valset)
    for fn in file_list:
        try:
            lb = fn.split('.')[0].split('_')[1].upper()
            assert len(lb)==4
            assert '^' not in lb
            fp = os.path.join(valset, fn)
            im = Image.open(fp)
            ima = np.array(im.convert('L'))
            inxx = [toks.index(v) for v in lb]
            strarr += [lb]
            valarr += inxx
            imarr.append(np.expand_dims(ima, axis=-1))
            ixarr += [[len(imarr) - 1, i] for i in range(4)]
        except:
            pass
    spar_lb = tf.SparseTensorValue(ixarr, valarr, [len(imarr), 4])
    return np.array(imarr), spar_lb, strarr


def gettrain():
    ixarr = []
    imarr = []
    valarr = []
    file_list = os.listdir(trainset)
    while True:
        try:
            if len(imarr) == batch_size:
                imarr = []
                ixarr = []
                valarr = []
            fn = random.choice(file_list)
            lb = fn.split('.')[0].split('_')[1].upper()
            assert len(lb) == 4
            assert '^' not in lb
            fp = os.path.join(trainset, fn)
            im = Image.open(fp)
            ima = np.array(im.convert('L'))
            ixx = [toks.index(v) for v in lb]
            valarr += ixx
            imarr.append(np.expand_dims(ima, axis=-1))
            ixarr += [[len(imarr) - 1, i] for i in range(4)]
            if len(imarr) == batch_size:
                spar_lb = tf.SparseTensorValue(ixarr, valarr, [batch_size, 4])
                yield np.array(imarr), spar_lb
                imarr = []
                ixarr = []
                valarr = []
                #file_list = os.listdir(trainset)
        except:
            pass
            #traceback.print_exc()

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

def processValidationImg():
    file_list = os.listdir(r'D:\Users\yl_gong\Downloads\images\images')
    for fn in file_list:
        try:
            fp = path.join(r'D:\Users\yl_gong\Downloads\images\images', fn)
            label = fn.split('.')[0].split('_')[1].upper()
            if len(label)!=4:
                raise Exception("");
            im = processim(fp)
            cv2.imwrite(path.join('xxx', uuid.uuid4().hex + '_' + label + '.jpg'), im)
        except:
            pass


if __name__=='__main__':
    processValidationImg()