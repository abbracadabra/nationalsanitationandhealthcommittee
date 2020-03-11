from config import *
from model import *
from util import *

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,modelfile)

if __name__=='__main__':
    im = np.array([processim(r'D:\Users\xxx\Desktop\下载.gif')[:,:,[1]]])
    predices = sess.run([argmax_logits], feed_dict={images: im})
    print(''.join([vb[i] for i in predices[0][0]]))