import tensorflow as tf
from config import *

images = tf.placeholder(dtype=tf.float32,shape=[None,40,135,1],name='image_holder')#[None,40,135,1]
sparse_label = tf.sparse.placeholder(tf.int32)
_ = images/255.
_ = tf.layers.conv2d(inputs=_,filters=32,kernel_size=3,strides=(2, 2),padding="SAME")#[None,20,68,64]
_ = tf.layers.batch_normalization(_)
_ = tf.nn.leaky_relu(_)
_ = tf.layers.dropout(_,0.7)
_ = tf.layers.conv2d(inputs=_,filters=32,kernel_size=3,strides=(2, 2),padding="SAME")#[None,10,34,64]
_ = tf.layers.batch_normalization(_)
_ = tf.nn.leaky_relu(_)
_ = tf.layers.dropout(_,0.7)
_ = tf.layers.conv2d(inputs=_,filters=32,kernel_size=3,strides=(2, 2),padding="SAME")#[None,5,17,64]
_ = tf.layers.batch_normalization(_)
_ = tf.nn.leaky_relu(_)
_ = tf.layers.dropout(_,0.7)
_ = tf.layers.conv2d(inputs=_,filters=32,kernel_size=3,strides=(2, 2),padding="SAME")#[None,3,9,64]
_ = tf.layers.batch_normalization(_)
_ = tf.nn.leaky_relu(_)
_ = tf.layers.dropout(_,0.7)
_ = tf.layers.conv2d(inputs=_,filters=32,kernel_size=3,strides=(3, 1),padding="VALID")#[None,1,9,64]
_ = tf.layers.batch_normalization(_)
_ = tf.nn.leaky_relu(_)
_ = tf.layers.dropout(_,0.7)
_ = tf.squeeze(_,axis=1)#[None,9,64]
time_step = tf.shape(_)[1]#9
seq_len = tf.tile(tf.expand_dims(time_step,axis=0),[tf.shape(_)[0]])#[9]*batchsize

# lstmcell = tf.nn.rnn_cell.LSTMCell(num_units=64)
# initial_state = lstmcell.zero_state(batch_size, dtype=tf.float32)
# outputs, _ = tf.nn.dynamic_rnn(cell=lstmcell,inputs=_,sequence_length=seq_len,initial_state=initial_state)#[None,35,64]
logits = tf.layers.dense(_, units = num_chars)#[None,9,num_chars]
logits_timemajor = tf.transpose(logits,[1,0,2])#[9,None,num_chars]
prob = tf.nn.softmax(logits,axis=-1)#[None,24,num_chars]
loss = tf.reduce_mean(tf.nn.ctc_loss(sparse_label, inputs=logits_timemajor, sequence_length=seq_len))

decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=logits_timemajor,sequence_length=seq_len,merge_repeated=False,beam_width=1)#top_n*[[batch_size, max_decoded_length]]
dense_decodes = tf.sparse_to_dense(decodes[0].indices,decodes[0].dense_shape,decodes[0].values,default_value=-1,name='ctc_best_path')

#tf.summary.scalar("ctcloss",loss)
#logging = tf.summary.merge_all()
#writer = tf.summary.FileWriter(log_dir)

