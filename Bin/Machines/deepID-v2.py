from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
# from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python import SKCompat
from .DataRepresentation import LoadData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tf.logging.set_verbosity(tf.logging.INFO)

# Convolutional neural net contrusction function
# It describes every layer separetly
 
def cnn_model_fn(features, labels, mode):
    # Input Layer, standard square 31*31 gray 
    input_layer = tf.reshape(features, [-1, 31, 31, 1])
 
    # Convolutional Layer #1, takes as the input the input_layer, valid padding and the AF is a ReLU
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=20,
        kernel_size=[4, 4],
        padding="valid",
        activation=tf.nn.relu
    )     
    # Normalization Layer #1, experimental normalization layer just after the conv layer
    # norm1 = tf.layers.local_response_normalization(
    #   inputs=conv1,
    #   depth_radius=5,
    #   alpha=0.0001,
    #   beta=0.75
    #)
 

    # Pooling Layer #1, using maxplooling takes as an input the conv1 layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2, takes as an input the output of the maxplooling1, 40 filters
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=40,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu
    )
    
    # Normalization Layer #2
    # norm2 = tf.layers.local_response_normalization(
    #   inputs=conv2,
    #   depth_radius=5,
    #   alpha=0.0001,
    #   beta=0.75
    #)
 
    # Pooling Layer #2, takes as an input the output of the 2nd convonlutional layer 
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3, takes as the input the output of the 2nd max_pooling layer, augumenting the dimensions to 60
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=60,
        kernel_size=[3, 3],
        padding='valid',
        activation=tf.nn.relu
    )
    
    # Pooling Layer #3, finaly pooling layer
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    # Convolutional Layer #4, final convolution layer, the output has 80 in depth
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=80,
        kernel_size=[2, 2],
        padding='valid',
        activation=tf.nn.relu
    )
 
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 1, 2, 80]
    # Output Tensor Shape: [batch_size, 1 * 2 * 80]
    conv4_flat = tf.reshape(conv4, [-1, 80])
    pool3_flat = tf.reshape(pool3, [-1, 80])
     
    # Experimental
    
     # Concat conv4 and pool3
    # conc = tf.concat(['conv4_flat','pool3_flat'], axis = -1)
    
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 1 * 2 * 80]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=conv4_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that the element will be kept
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.6,
        training=mode == learn.ModeKeys.TRAIN
    )

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)
    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
i            logits=logits
        )

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD"
        )

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
                   input=logits, axis=1),
        "probabilities": tf.nn.softmax(
                   logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)

   
#Building the classifier based on the model function that we've created above in the cnn_model_fn
classifier = learn.Estimator(
                   model_fn=cnn_model_fn,
                   model_dir="../Data/Models"
                   )

# tensors_to_log = {"probabilities": "softmax_tensor"}
# logging_hook = tf.train.LoggingTensorHook(
#                tensors=tensors_to_log, every_n_iter=50)


# Defining the Training function that will fit the model by using the classifier that we made
def Training():
    print("[+] Welcome to the training program.")
    X, Y = LoadData()
    train_data, eval_data, train_labels, eval_labels = train_test_split(X, Y, test_size=0.2, random_state=42)
    # print(type(train_labels))
    global classifier
    # print("[+] till now working")
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    # global logging_hook 
    # print("[+] till now working")

    # Train the model
    classifier.fit(x=train_data,y=train_labels, batch_size= 15, steps= 10000)
    # print("[+] till now working")
    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy": learn.MetricSpec(
                          metric_fn=tf.metrics.accuracy,
                          prediction_key="classes"),
                  }

    # Evaluate the model and print results
    eval_results = classifier.evaluate(
                              x=eval_data,
                              y=eval_labels,
                              metrics=metrics
                  )
    print(eval_results)


# The function that will use the trained model to predict the labels    
def Solver(Input):
    global classifier
    os.chdir('Bin')
    # predictions = classifier.predict(x=Input, as_iterable=True)
    for i, p in enumerate(predictions):
        return p['classes']
     
# In case of wrong classification, the intervenience of the human oracle to correct the labeling     
def Correction(Input):
    global classifier
    X_train = np.empty((15,31,31),dtype=np.float32)
    Y_train = np.empty((15))
    cor = int(raw_input('What is the correct answer ?\n-->'))
    for i in xrange(15):
        X_train[i] = Input
        Y_train[i] = cor  
    classifier.fit(x=X_train, y=Y_train, batch_size=7, steps=3000, monitors=[logging_hook])

    
__all__ = ["Training","Solver","Correction"]
