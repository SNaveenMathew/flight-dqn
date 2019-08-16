import numpy as np
import math
import sys
sys.path.append('../gym-flight/')
import gym_flight
from gym_flight.utils.numpy_util import Sparse3dArray
from full3d_util import get_action_discrete, get_action_continuous, fix_XY, train_test_split

def get_space_around_ac(ts_mat, aircraft_x, aircraft_y, aircraft_z, half_x_length, half_y_length, half_z_length, x_length, y_length, z_length, state_dim):
    lower_x = aircraft_x - half_x_length
    upper_x = aircraft_x + half_x_length + 1
    lower_y = aircraft_y - half_y_length
    upper_y = aircraft_y + half_y_length + 1
    lower_z = aircraft_z - half_z_length
    upper_z = aircraft_z + half_z_length + 1
    aircraft_space = ts_mat[max(0, lower_x):min(x_length, upper_x), max(0, lower_y):min(y_length, upper_y), max(0, lower_z):min(z_length, upper_z)]
    aircraft_space = pad_zeros(aircraft_space, lower_x, lower_y, lower_z, upper_x, upper_y, upper_z, x_length, y_length, z_length)
    if aircraft_space.shape != tuple(state_dim):
        print(aircraft_space.shape)
    
    aircraft_space[half_x_length, half_y_length, half_z_length] = -1
    return aircraft_space


# For each timestamp of controlled aircraft the get_env_action_aircraft function computes the surrounding space around aircraft (feature space) and the future action (output)
def get_env_action_aircraft(controlled_ac_df, space_X, half_x_length, half_y_length, half_z_length, x_length, y_length, z_length, state_dim, env = True, discrete_action = True):
    uniq_ts = controlled_ac_df['ts'].unique()
    uniq_ts.sort()
    uniq_ts = uniq_ts[:-1]
    space_full_ts = [space_X[ts-1] for ts in uniq_ts]

    if discrete_action:
        actions = np.zeros(len(uniq_ts))
    else:
        actions = np.zeros((len(uniq_ts), 3))
    # The following step can be done without loop by creating lag variable. It will be fixed later
    for i, ts in enumerate(uniq_ts):
        ts_mat = space_full_ts[i].toarray()
        controlled_ac_ts_dat = controlled_ac_df[controlled_ac_df['ts'] == ts]
        aircraft_x = controlled_ac_df['x'].iloc[0]
        aircraft_y = controlled_ac_df['y'].iloc[0]
        aircraft_z = controlled_ac_df['z'].iloc[0]
        space_around_ac = get_space_around_ac(ts_mat, aircraft_x, aircraft_y, aircraft_z, half_x_length, half_y_length, half_z_length, x_length, y_length, z_length, state_dim)
        if discrete_action:
            actions[i] = get_action_discrete(controlled_ac_df, i)
        else:
            actions[i, :] = get_action_continuous(controlled_ac_df, i)

    return([[space_full_ts], [actions]])


# This functions calls get_env_action_aircraft for multiple controlled aircraft taken one at a time
def get_X_Y(uniq_id, flight_data, space_X, half_x_length, half_y_length, half_z_length, x_length, y_length, z_length, state_dim, env = True, discrete_action = True):
    X = Y = []
    for i, id in enumerate(uniq_id):
        print(i)
        controlled_ac_df = flight_data[flight_data['id'] == id]
        x, y = get_env_action_aircraft(controlled_ac_df, space_X, half_x_length, half_y_length, half_z_length, x_length, y_length, z_length, state_dim, env = True, discrete_action = discrete_action)
        X.append(x)
        Y.append(y)

    return [X, Y]



def pad_zeros(aircraft_space, lower_x, lower_y, lower_z, upper_x, upper_y, upper_z, x_length, y_length, z_length):
    if lower_x < 0:
        aircraft_space = pad_zeros_left(aircraft_space, -lower_x, axis = 0)
    if lower_y < 0:
        aircraft_space = pad_zeros_left(aircraft_space, -lower_y, axis = 1)
    if lower_z < 0:
        aircraft_space = pad_zeros_left(aircraft_space, -lower_z, axis = 2)
    if upper_x > x_length:
        aircraft_space = pad_zeros_right(aircraft_space, upper_x-x_length, axis = 0)
    if upper_y > y_length:
        aircraft_space = pad_zeros_right(aircraft_space, upper_y-y_length, axis = 1)
    if upper_z > z_length:
        aircraft_space = pad_zeros_right(aircraft_space, upper_z-z_length, axis = 2)
    return aircraft_space

def pad_zeros_left(np_array, num_zeros, axis):
    shp = np_array.shape
    out_array = np.zeros((shp[0] + num_zeros*(axis == 0), shp[1] + num_zeros*(axis == 1), shp[2] + num_zeros*(axis == 2)))
    out_array[num_zeros*(axis == 0):, num_zeros*(axis == 1):, num_zeros*(axis == 2):] = np_array
    return out_array

def pad_zeros_right(np_array, num_zeros, axis):
    shp = np_array.shape
    out_array = np.zeros((shp[0] + num_zeros*(axis == 0), shp[1] + num_zeros*(axis == 1), shp[2] + num_zeros*(axis == 2)))
    if axis == 0:
        out_array[:(-num_zeros), :, :] = np_array
    elif axis == 1:
        out_array[:, :(-num_zeros), :] = np_array
    else:
        out_array[:, :, :(-num_zeros)] = np_array
    return out_array

