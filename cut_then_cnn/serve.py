import tensorflow as tf
from config import *

saver = tf.train.import_meta_graph(modelfile+'.meta')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, modelfile)

builder = tf.saved_model.Builder("expmdl")

builder.add_meta_graph_and_variables(sess,['serve'])

builder.save()