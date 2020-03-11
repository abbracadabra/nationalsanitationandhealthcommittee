from config import *
from model import *
import tensorflow as tf
import time
import sys



images = tf.placeholder(dtype=tf.float32,shape=[None,40,135,1],name='image_holder')#[None,40,135,1]
label = tf.placeholder(dtype=tf.int32,shape=[None,4])#[None,4]
onehot_lb = tf.one_hot(label,depth=num_chars)#[None,4,36]
_ = images/255.
_ = tf.layers.conv2d(inputs=_,filters=64,kernel_size=(7,7),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
_ = tf.layers.max_pooling2d(_,pool_size=(4,4),strides=(4,4),padding='SAME')
_ = tf.layers.dropout(_,rate=0.9)
_ = tf.layers.batch_normalization(_)

_ = tf.layers.conv2d(inputs=_,filters=64,kernel_size=(5,5),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
_ = tf.layers.max_pooling2d(_,pool_size=(2,2),strides=(2,2),padding='SAME')
_ = tf.layers.dropout(_,rate=0.9)
_ = tf.layers.batch_normalization(_)

_ = tf.layers.conv2d(inputs=_,filters=128,kernel_size=(3,3),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
_ = tf.layers.max_pooling2d(_,pool_size=(2,2),strides=(2,2),padding='SAME')
_ = tf.layers.dropout(_,rate=0.9)
_ = tf.layers.batch_normalization(_)

_ = tf.layers.flatten(_)
_ = tf.layers.dense(_,num_chars*4)
logits = tf.reshape(_,[-1,4,num_chars])#[none,4,36]
argmax_logits = tf.argmax(logits,axis=-1,output_type=tf.int32)#[None,4]
acc = tf.reduce_sum(tf.cast(tf.equal(tf.reduce_sum(tf.cast(tf.equal(label,argmax_logits),tf.int32),axis=-1,keepdims=False),4),tf.int32))/tf.shape(images)[0]
loss = tf.reduce_mean(-tf.log(tf.reduce_prod(tf.reduce_sum(tf.nn.softmax(logits,axis=-1)*onehot_lb,axis=-1,keepdims=False),axis=-1,keepdims=False)))


