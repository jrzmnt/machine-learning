import os
import sys
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
from sklearn.model_selection import train_test_split



def start_config(gpu):
        print('Configuring environment')

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        #tf.reset_default_graph()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def idm_data_transform(data):
    #print(data.shape)
    data_t = []
    for i in range(0, data.shape[0], 16):
        a = data[i][:,:,0:3].reshape(1,224,224,3)
        b = data[i][:,:,3:].reshape(1,224,224,3)
        c = np.concatenate((a,b), axis=0)
        data_t.append(c)
        
    return data_t


def cnn_model(x_train_data, keep_rate=0.7, seed=None):
    
    with tf.name_scope("layer_a"):
        # conv => 16*16*16
        conv1 = tf.layers.conv3d(inputs=x_train_data, filters=16, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # conv => 16*16*16
        conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # pool => 8*8*8
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
        
    with tf.name_scope("layer_c"):
        # conv => 8*8*8
        conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # conv => 8*8*8
        conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # pool => 4*4*4
        #pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2, 2, 2], strides=2)
        
        
    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=conv5, training=True)
        
    with tf.name_scope("fully_con"):
        flattening = tf.layers.flatten(cnn3d_bn)#tf.reshape(cnn3d_bn, [-1, 4*4*4*128])
        dense = tf.layers.dense(inputs=flattening, units=128, activation=tf.nn.relu)
        # (1-keep_rate) is the probability that the node will be kept
        dropout = tf.layers.dropout(inputs=dense, rate=keep_rate, training=True)
        
    with tf.name_scope("y_conv"):
        y_conv = tf.layers.dense(inputs=dropout, units=2)
    
    return y_conv


def train_neural_network(x_train_data, y_train_data, x_test_data, y_test_data, learning_rate=0.05, keep_rate=0.7, epochs=10, batch_size=32):

    gpu_number = 0
    start_config(gpu_number)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
    with tf.name_scope("cross_entropy"):
        prediction = cnn_model(x_input, keep_rate, seed=1)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input))
        saver = tf.train.Saver()
                              
    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
           
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    iterations = int(len(x_train_data)/batch_size) + 1
    
    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        import datetime

        start_time = datetime.datetime.now()

        iterations = int(len(x_train_data)/batch_size) + 1
        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Epoch', epoch, 'started', end='')
            epoch_loss = 0
            # mini batch
            
            for itr in range(iterations):
                
                mini_batch_x = x_train_data[itr*batch_size: (itr+1)*batch_size]
                mini_batch_y = y_train_data[itr*batch_size: (itr+1)*batch_size]
                

                #print('x:'+str(mini_batch_x.shape))
                #print('y:'+str(mini_batch_y.shape))
                _optimizer, _cost = sess.run([optimizer, cost], 
                                            feed_dict={x_input: mini_batch_x, y_input: mini_batch_y})
                epoch_loss += _cost

            acc = 0
            itrs = int(len(x_test_data)/batch_size) + 1
            for itr in range(itrs):
                #print('\n '+str(itr*batch_size)+":"+str((itr+1)*batch_size))
                mini_batch_x_test = x_test_data[itr*batch_size: (itr+1)*batch_size]
                mini_batch_y_test = y_test_data[itr*batch_size: (itr+1)*batch_size]
                acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})

            end_time_epoch = datetime.datetime.now()
            print(' Testing Set Accuracy:',acc/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))

        end_time = datetime.datetime.now()

        print('Time elapse: ', str(end_time - start_time))


if __name__ == "__main__":

    idm_pkl_path = "/A/juarez/pickle_idm.npz"
    idm_pkl = np.load(idm_pkl_path)

    images = idm_pkl['images']
    actions = idm_pkl['actions']

    x_train, x_test, y_train, y_test = train_test_split(images, actions, test_size=0.1, shuffle=False)

    print('\nIDM Train dataset:', len(x_train))
    print('Class 0 size:', np.count_nonzero(y_train == [0., 1.]) / 2)
    print('Class 1 size:', np.count_nonzero(y_train == [1., 0.]) / 2)

    print('\nIDM Valid dataset:', len(x_test))
    print('Class 0 size:', np.count_nonzero(y_test == [0., 1.]) / 2)
    print('Class 1 size:', np.count_nonzero(y_test == [1., 0.]) / 2, '\n')

    x_train = np.asarray(idm_data_transform(x_train))
    print("x_train.shape:"+str(x_train.shape))

    x_test = np.asarray(idm_data_transform(x_test))
    print("x_test.shape:"+str(x_test.shape))

    n_classes = 2

    with tf.name_scope('inputs'):
        x_input = tf.placeholder(tf.float32, shape=[None, 2, 224, 224, 3])
        y_input = tf.placeholder(tf.float32, shape=[None, n_classes]) 
        train_neural_network(x_train[:80], y_train[:80], x_test[:30], y_test[:30], epochs=5, batch_size=100, learning_rate=0.001)