import tensorflow as tf
import sys
sys.path.append('../gym-flight/')
import gym_flight
from gym_flight.utils.numpy_util import Sparse3dArray
import numpy as np

class CNN_multiclass:
    def __init__(self, state_dim, time_dim, action_size, learning_rate, conv1_shape = (32, 8, 4), conv2_shape = (64, 4, 2), conv3_shape = (128, 3, 2)):
        self.target_ = tf.placeholder(tf.int32, [None], name='target')
        self.state_ = tf.placeholder(tf.float32, [None] + state_dim + [time_dim], name='state')
        self.learning_rate = learning_rate
        
        # conv1: (, 32, 8, 4)
        self.conv1 = tf.layers.conv3d(self.state_, conv1_shape[0], conv1_shape[1], conv1_shape[2], activation=tf.nn.relu, name='conv1')
        
        # conv2: (, 64, 4, 2)
        self.conv2 = tf.layers.conv3d(self.conv1, conv2_shape[0], conv2_shape[1], conv2_shape[2], activation=tf.nn.relu, name='conv2')
        
        # conv3: (, 128, 3, 2)
        self.conv3 = tf.layers.conv3d(self.conv2, conv3_shape[0], conv3_shape[1], conv3_shape[2], activation=tf.nn.relu, name='conv3')
        
        # flat: (, 768)
        self.flat = tf.layers.flatten(self.conv3, name='flat')
        
        # fc1: (, 1024)
        self.fc1 = tf.layers.dense(self.flat, 1024, activation=tf.nn.relu, name='fc1')
        
        # Fc2 (b, 128) - may be bottleneck layer because action size is larger
        self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.relu, name='fc2')
        
        # Prediction (b, action_size): output probabilities
        self.output = tf.layers.dense(self.fc2, action_size, name='output')
        
        # Probabilities
        self.exp_output = tf.exp(self.output)
        self.row_sums = tf.reduce_sum(self.exp_output, axis = 1)
        self.probabilities = self.exp_output/tf.reshape(self.row_sums, (-1, 1))
        
        # Loss: multiclass cross entropy
        # self.loss = tf.losses.softmax_cross_entropy(onehot_labels = tf.one_hot(self.target, action_size), logits = self.output)
#         self.loss = tf.losses.softmax_cross_entropy(onehot_labels = tf.one_hot(self.target_, action_size), logits = self.output)
        self.loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(self.target_, action_size) * tf.log(self.probabilities), reduction_indices=[1]))
        
        # Optimizer: Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        
        # Train Op
        self.train = self.optimizer.minimize(self.loss)



# Function for training the network
def train(sess, cnn, state, action):
    feed_dict = {cnn.state_: state_to_frame(state), cnn.target_: action}
    loss, _ = sess.run([cnn.loss, cnn.train], feed_dict=feed_dict)
    return loss


# Function to convert Sparse3dArray states to np.array
def state_to_frame(state):
    """
    Convert states to frames
    Background:
    Since our state space (100, 100, 100) or (100, 100, 100, num_frames) is a rather large sparse space, storing as
    numpy array format would be infeasible due to heavy memory usage when constructing replay memory. Therefore, we use
    Sparse3dArray object to store states and only convert it back to numpy array when feeding into network.
    Naming conventions:
    States are states storing in Sparse3dArray format
    Frames are states converted back to numpy arrays
    """
    # If it is a single state (i.e. Sparse3dArray)
    if isinstance(state, Sparse3dArray):
        state_array = state.toarray()
        output = np.zeros(state_array.shape + (1,))
        output[:, :, :, 0] = state_array
        return output
    # If it is a stacked state (i.e. list of Sparse3dArray)
    elif all(isinstance(item, Sparse3dArray) for item in state):
        return np.stack([state_to_frame(item) for item in state], 0)
    # If it is a list of stacked state (i.e. list of list of Sparse3dArray)
    elif all(isinstance(item, Sparse3dArray) for items in state for item in items):
        return np.array([np.stack([item.toarray() for item in items], -1) for items in state])
    else:
        raise ValueError('Expected Sparse3dArray or List[Sparse3dArray] or List[List[Sparse3dArray]], instead got {}'
                         .format(type(state)))



def get_logits(sess, cnn, state):
    """
    Get predicted probabilities of actions for given state
    """
    logits = sess.run(cnn.output, feed_dict={cnn.state_: state_to_frame(state)})
    return logits



def get_loss(sess, cnn, state, target):
    """
    Get action from given state using ε-greedy
    """
    loss = sess.run(cnn.loss, feed_dict={cnn.state_: state_to_frame(state), cnn.target_: target})
    return loss



# Probability is a monotonous increasing function of output. Therefore, argmax(probability) can be replaced with argmax(output).
def get_predicted_class(sess, cnn, state):
    output = sess.run(cnn.output, feed_dict={cnn.state_: state_to_frame(state)})
    classes = np.argmax(output, axis = 1)
    return classes