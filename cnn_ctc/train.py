from config import *
from model import *
import tensorflow as tf
import time
import sys
from util import *
from model import *

saver = tf.train.Saver()
ops = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,model_dir)

if __name__=='__main__':
    val_x, val_label, val_strs = getval()
    val_history = []
    val_acc_max = 0;
    val_best_pos = -1
    for j, (imgs, spar_lb) in enumerate(gettrain()):
        _,ctc_err, _logits = sess.run([ops, loss, logits], feed_dict={images: imgs, sparse_label: spar_lb})
        print(ctc_err)
        if int(time.time() % 50) == 0:
            val_loss,_preds = sess.run([loss,dense_decodes], feed_dict={images: val_x, sparse_label: val_label})  # top_n*[[batch_size, max_decoded_length]]
            print("valloss:"+str(val_loss))
            strarr = []
            for line in _preds:
                tmpstr = (''.join([toks[char_index] if char_index != -1 else '' for char_index in line])).replace('_','')
                strarr.append(tmpstr)
            val_acc = np.sum([1 if val_strs[i] == strarr[i] else 0 for i in range(len(val_strs))]) / len(val_strs)
            if val_acc > val_acc_max:
                print("best val acc:" + str(val_acc))
                val_acc_max = val_acc
                val_best_pos = len(val_history) - 1
                saver.save(sess, model_dir)
            elif len(val_history) - val_best_pos >30:
                sys.exit()
            # saver.save(sess,model_dir,write_meta_graph=False)




