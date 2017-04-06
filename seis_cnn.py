# 0, make the code correct
# 1, load data by python directly , show the result directly
# 2, split train and test codes, save the trained network
# 3, test for a whole portion
# 4, visualize the train progress and the network/dictionary
# 5, adjust the paramter
# 6, add more train samples
# 7, transfer to workstation
# ...


from __future__ import absolute_import
from __future__ import division
import seismic_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#initialization
tf.set_random_seed(0)
model_path = "../data/model.ckpt" 

K = 32  
L = 32


X = tf.placeholder(tf.float32,[None,28,14,1])
Y_ = tf.placeholder(tf.float32,[None,28,14,1])


#model

pkeep = 1.0

#Y1  = tf.nn.dropout(Y1f,pkeep)
W1 = tf.Variable(tf.truncated_normal([8, 4, 1, K] ,stddev=0.001))
B1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.truncated_normal([1, 1, K, L] ,stddev=0.001))
B2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.truncated_normal([8, 4, L, 1] ,stddev=0.001))
B3 = tf.Variable(tf.zeros([1]))



Y1 = tf.nn.relu(tf.nn.conv2d(X,  W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 1, 1, 1], padding='SAME') + B3)


#placeholder for correct answers
output = Y3

#loss function
cross_entropy = tf.reduce_sum(tf.pow(Y_-Y3,2))
# % of correct answers found in batch

#optimizer = tf.train.GradientDescentOptimizer(0.001)
#train_step= optimizer.minimize(cross_entropy)

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

ss = seismic_data.load_data()
saver = tf.train.Saver()


nt = 500
c  = np.zeros(nt)

def training_step():    
    for i in range(nt):
        # load batch of images and correct answers
        batch_X, batch_Y = ss.next_batch(100)
        #print batch_X.shape, batch_Y.shape
        train_data = {X:batch_X, Y_:batch_Y}

        # train
        sess.run(train_step,feed_dict=train_data)
        #success?
        ct = sess.run([cross_entropy], feed_dict=train_data)
        c[i] = ct[0]
        print(i,c[i])

def test_step():       
    t_X, t_Y = ss.test_data()
    test_data2 = {X:t_X, Y_:t_Y}
    Y_o = sess.run(output, feed_dict = test_data2)
    t_X = t_X.squeeze()
    t_Y = t_Y.squeeze()
    Y_o = Y_o.squeeze()
    
    # plot results
    plt.subplot(1,3,1)
    plt.imshow(t_X)
    plt.title('Input')
    plt.subplot(1,3,2)
    plt.imshow(t_Y)
    plt.title('Original')
    plt.subplot(1,3,3)
    plt.imshow(Y_o)
    plt.title('Output')
#    plt.figure(2)
#    plt.plot(c)

#training_step()
#saver.save(sess, model_path)
#saver.restore(sess, model_path)
test_step()



