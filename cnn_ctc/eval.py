from config import *
from model import *
from util import *

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,model_dir)


if __name__=='__main__':
    im = np.array([processim(r'D:\Users\xxx\Desktop\下载.gif')[:,:,[1]]])
    _preds = sess.run([dense_decodes], feed_dict={images: im})
    strarr=[]
    for line in _preds:
        tmpstr = (''.join([toks[char_index] if char_index != -1 else '' for char_index in line[0]])).replace('_', '')
        strarr.append(tmpstr)
    print(strarr)


