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


def toNp(image):
    tp = type(image)
    if 'PIL' in str(tp):
        return np.uint8(image)
    return image

def toPIL(image):
    tp = type(image)
    if 'numpy' in str(tp):
        return Image.fromarray(image)
    return image

def randomshift(im):
    im = toPIL(im)
    bgwidth = 24
    bgheight = 40
    bg = Image.new('L', (bgwidth, bgheight), color=255)
    bg = addBlackDots(bg)
    bg = toPIL(bg)
    #bg = addLines(bg)
    #bg = addWhiteDots(bg);
    bg.paste(im,(np.random.randint(-3, 4),np.random.randint(-4, 4)))
    return bg



def addBlackDots(image):
    image = toNp(image)
    image.setflags(write=1)
    s_vs_p = 0.5
    amount = np.random.rand()/30
    #out = np.copy(image)
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper))
              for i in image.shape]
    image[coords] = 0
    return image

def addWhiteDots(image):
    image = toNp(image)
    image.setflags(write=1)
    s_vs_p = 0.5
    amount = np.random.rand()/6
    #out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt))
              for i in image.shape]
    image[coords] = 255
    return image

def addLines(img):
    img = toPIL(img)
    width, height = img.size
    for i in range(2):
        lineWidth = np.random.choice([0, 1, 2, 3], p=[0.5, 0.2, 0.2, 0.1])
        if lineWidth != 0:
            draw = ImageDraw.Draw(img)
            draw.line((randint(0, width), randint(0, height), randint(0, width), randint(0, height)), fill=0,
                      width=lineWidth)
    return img


def getSimData():
    x = []
    lb=[]
    bgwidth = 24
    bgheight = 40
    file_list = os.listdir(letterdir)
    while True:
        fn = random.choice(file_list)
        c = fn[0]
        fp = os.path.join(letterdir,fn)
        fg = Image.open(fp).convert('L')
        fgwidth, fgheight = fg.size
        bg = Image.new('L', (bgwidth, bgheight),color=255)
        bg = addLines(bg)
        bg = addWhiteDots(bg);
        bg = Image.fromarray(bg)
        bg.paste(fg, (np.random.randint(0, max(bgwidth-fgwidth,1)), np.random.randint(0, max(bgheight-fgheight,1))))
        bg = addWhiteDots(bg)
        bg = addBlackDots(bg)
        #cv2.imwrite(path.join('tmp',uuid.uuid4().hex + '.jpg'), bg)
        bg = np.array(bg)
        bg = np.expand_dims(bg, axis=-1)
        x.append(bg)
        lb.append(vb.index(c))
        if len(x) == batchsize:
            yield x,lb
            x=[]
            lb=[]


def getRealData():
    file_list = os.listdir(trainset)
    x = []
    lb = []
    while True:
        fn = random.choice(file_list)
        c = fn.split('.')[0].split('_')[1].upper()
        if c in vb:
            lb.append(vb.index(c))
            fp = os.path.join(trainset, fn)
            im = Image.open(fp).convert('L')
            im = randomshift(im)
            im = addWhiteDots(im)
            im = addBlackDots(im)
            im = np.array(im)
            im = np.expand_dims(im, axis=-1)
            x.append(im)
        if len(x) == batchsize:
            yield x, lb
            x = []
            lb = []

def getRealVal():
    file_list = os.listdir(valset)
    x = []
    lb = []
    for fn in file_list:
        c = fn.split('.')[0].split('_')[1].upper()
        if c in vb:
            lb.append(vb.index(c))
            fp = os.path.join(valset, fn)
            im = Image.open(fp).convert('L')
            im = np.array(im)
            im = np.expand_dims(im, axis=-1)
            x.append(im)
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

def processValidationImg():
    file_list = os.listdir(valset)
    x = []
    lb = []
    for fn in file_list:
        try:
            fp = path.join(valset, fn)
            label = fn.split('.')[0].split('_')[1].upper()
            im = processim(fp)

            # cv2.imwrite(path.join('xxxx', uuid.uuid4().hex + '_' + label[0] + '.jpg'), im[:, 20:44, :])
            # cv2.imwrite(path.join('xxxx', uuid.uuid4().hex + '_' + label[1] + '.jpg'), im[:, 44:68, :])
            # cv2.imwrite(path.join('xxxx', uuid.uuid4().hex + '_' + label[2] + '.jpg'), im[:, 68:92, :])
            # cv2.imwrite(path.join('xxxx', uuid.uuid4().hex + '_' + label[3] + '.jpg'), im[:, 92:116, :])
        except:
            pass

        x+=[im[:,20:44,[0]],im[:,44:68,[0]],im[:,68:92,[0]],im[:,92:116,[0]]]
        lb+=[vb.index(label[0]),vb.index(label[1]),vb.index(label[2]),vb.index(label[3])]
    return x,lb

processValidationImg()