import tensorflow as tf
import sys
sys.path.append('../gym-flight/')
import gym_flight
from gym_flight.utils.numpy_util import Sparse3dArray
import numpy as np

class CNN_continuous:
    def __init__(self, state_dim, learning_rate):
        self.target_ = tf.placeholder(tf.int32, [None, 3], name='target')
        self.state_ = tf.placeholder(tf.float32, [None] + state_dim + [1], name='state')
        self.learning_rate = learning_rate
        
        # conv1: (, 32, 8, 4)
        self.conv1 = tf.layers.conv3d(self.state_, 32, 8, 4, activation=tf.nn.relu, name='conv1')
        
        # conv2: (, 64, 4, 2)
        self.conv2 = tf.layers.conv3d(self.conv1, 64, 4, 2, activation=tf.nn.relu, name='conv2')
        
        # conv3: (, 128, 3, 2)
        self.conv3 = tf.layers.conv3d(self.conv2, 128, 3, 2, activation=tf.nn.relu, name='conv3')
        
        # flat: (, 768)
        self.flat = tf.layers.flatten(self.conv3, name='flat')
        
        # fc1: (, 1024)
        self.fc1 = tf.layers.dense(self.flat, 1024, activation=tf.nn.relu, name='fc1')
        
        # Fc2 (b, 128) - may be bottleneck layer because action size is larger
        self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.relu, name='fc2')
        
        # Split into 3 linear output units for:
        # - change in speed
        # - change in altitude
        # - change in heading
        # This should probably be changed to separate dense 'heads' that connect to individual outputs
        # If we maintain different dense heads, if one output is noisy, then the weights of that dense head will be random (with the hope that weights of other heads are not affected)
        self.output = tf.layers.dense(self.fc2, 3, name='output')
        
        self.loss = tf.losses.mean_squared_error(labels = self.target_, predictions = self.output)
        
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



def get_predicted_action(sess, cnn, state):
    output = sess.run(cnn.output, feed_dict={cnn.state_: state_to_frame(state)})
    return output



def get_loss(sess, cnn, state, action):
    output = sess.run(cnn.loss, feed_dict={cnn.state_: state_to_frame(state), cnn.target_: action})
    return output
