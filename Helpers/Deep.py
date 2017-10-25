from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)


# CNN Model Function

def cnn_model_fn(features, labels, mode):

# Input Layer

    input_layer = tf.reshape(features, [-1, 62, 62, 1])


# Convolutional Layer #1

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=20,
        kernel_size=[4, 4],
        padding="valid",
        activation=tf.nn.relu
    )

# Normalization Layer #1
    norm1 = tf.layers.local_response_normalization(
        inputs=conv1,
        depth_radius=5,
        alpha=0.0001,
        beta=0.75
    )


# Pooling Layer #1

    pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=40,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu
    )

#Normalization Layer #2
    norm2 = tf.layers.local_response_normalization(
       inputs=conv2,
       depth_radius=5,
       alpha=0.0001,
       beta=0.75
   )

# Pooling Layer #2
  
    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

# Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=60,
        kernel_size=[3, 3],
        padding='valid',
        activation=tf.nn.relu
    )
   
  
# Pooling Layer #3
  
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# Convolutional Layer #4

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
    conv4_flat = tf.reshape(conv4, [-1, 1 * 2 * 80])
    pool3_flat = tf.reshape(pool3, [-1, 1 * 2 * 60])

# Concat conv4 and pool3
 #   con = tf.concat(['conv4_flat','pool3_flat'], axis = -1)
# Dense Layer
# Densely connected layer with 1024 neurons
# Input Tensor Shape: [batch_size, 1 * 2 * 80]
# Output Tensor Shape: [batch_size, 1024]

    dense = tf.layers.dense(inputs=conv4_flat, units=1024, activation=tf.nn.relu)

# Add dropout operation; 0.6 probability that element will be kept

    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
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
            logits=logits
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
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    print("[+] Welcome to the training program.")
    train_data = np.load('../Data/CSV/X_train.npy')
    train_data.reshape(train_data.shape[0], 62, 62, 1)
    train_labels = np.load('../Data/CSV/Y_train.npy')
    eval_data = np.load('../Data/CSV/X_test.npy')
    eval_data.reshape(eval_data.shape[0], 62, 62, 1)
    eval_labels = np.load('../Data/CSV/Y_test.npy')
    # print(train_data.shape())
    # print(train_labels.shape())

    lfw_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="Data/Models")
    print("[+] till now working")
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    print("[+] till now working")

    # Train the model
    lfw_classifier.fit(x=train_data, y=train_labels, batch_size=5, max_steps=20000, monitors=[logging_hook])
    print("[+] till now working")
    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = lfw_classifier.evaluate(
        x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
