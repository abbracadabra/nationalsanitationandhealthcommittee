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

input = tf.placeholder(dtype='float32', shape=(None,40,24,1), name='input')
input2 = input/255.
label = tf.placeholder(dtype=tf.int64, shape=(None), name='label')
label2 = tf.one_hot(label,36)
temp = tf.layers.conv2d(inputs=input2,filters=64,kernel_size=(3,3),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
temp = tf.layers.conv2d(inputs=temp,filters=64,kernel_size=(3,3),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
temp = tf.layers.dropout(temp,0.2)
temp = tf.layers.conv2d(inputs=temp,filters=64,kernel_size=(3,3),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
temp = tf.layers.conv2d(inputs=temp,filters=64,kernel_size=(3,3),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
temp = tf.layers.dropout(temp,0.2)
temp = tf.layers.flatten(temp)
temp = tf.layers.dense(temp,36)
prob = tf.nn.softmax(temp)
predix = tf.argmax(prob,name='pred',axis=-1)
acc = tf.reduce_sum(tf.cast(tf.math.equal(tf.squeeze(tf.argmax(prob,axis=-1)),label),dtype=tf.int32))/tf.shape(label)[0]
acc = tf.identity(acc,name='acc')
loss = tf.losses.softmax_cross_entropy(label2,temp)
