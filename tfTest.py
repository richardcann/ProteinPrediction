# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:05:26 2018

@author: rmcp1g15
Get the data into objects with sequence, secondary structure
-create windowing with random samples of size= n, then output the x where each
amino acid in the window sequence has a 1, then the corresponding output (1, 0, 0) for Y
one hidden layer
get the weights and biases
get prediction, cost, optimizer.
Code template taken from: https://github.com/aymericdamien/TensorFlow-Examples

"""

from __future__ import print_function

from predictStruct import getStructs

import tensorflow as tf
import random
import numpy as np

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1
windowSize = 15

#number of hidden neurons and custom window size
n_hidden_1 = 32
n_input = 22*windowSize 
n_classes = 3
allStructs = getStructs()

#get 90% of data for training
training_size = len(allStructs)*.90
train_structs = []
test_structs =[]
for i in range(len(allStructs)):
    if i <= training_size:
        train_structs.append(allStructs[i])
    else:
        test_structs.append(allStructs[i])

#data inputs
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}



def multilayer_perceptron(x):
    #only one hidden layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


logits = multilayer_perceptron(X)

#softmas cross entropy loss func
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
#try to to sigmoid as loss
y_conv = tf.nn.sigmoid(logits)
loss = -(Y * tf.log(y_conv + 1e-12) + (1 - Y) * tf.log( 1 - y_conv + 1e-12))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = random.sample(range(len(train_structs)), batch_size)
        # Loop over all batches
        for i in total_batch:
            batch_x, batch_y = train_structs[i].getRepresentation(windowSize)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c /batch_size
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost = {:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    predictions = tf.argmax(pred,1)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #get all tetsing (1000 sequences)
    test_x, test_y = test_structs[0].getRepresentation(windowSize)
    for i in range(1, 1000):
        x, y = test_structs[i].getRepresentation(windowSize)
        test_x = np.concatenate((test_x, x))
        test_y = np.concatenate((test_y, y))
    #will get predictions of classes based on tests
    y_pred = predictions.eval(feed_dict={X: test_x})
    #compare with actual values to get accuracy
    print("Accuracy:", accuracy.eval(feed_dict={X: test_x, Y: test_y}))