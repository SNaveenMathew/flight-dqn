import numpy as np
import math
import sys
sys.path.append('../gym-flight/')
import gym_flight
from gym_flight.utils.numpy_util import Sparse3dArray

# Using the last known position of an aircraft within given timestamp range

def get_last_ts_data(ts, flight_data):
    last_position_data = flight_data[flight_data['ts'] <= ts].groupby(['id']).last()
    return last_position_data




# Creating lon * lat * altitude = 100 * 100 * 100 space at a particular timestamp. Each cell returns the total number of aircraft in (lon, lat, altitude)

def create_space_ts(ts, flight_data, x_length, y_length, z_length, with_ts_range = True, ts_range = None):
    if with_ts_range:
        lower_ts = ts - ts_range
    else:
        lower_ts = ts
    
    output_array = Sparse3dArray(shape = (x_length, y_length, z_length), dtype = np.int16)
    flight_data = flight_data[(flight_data['ts'] >= (lower_ts)) & (flight_data['ts'] <= ts)]
    
#     subtract_idx = ts - time_dim + 1
    unique_ts = flight_data['ts'].unique()
    
    if flight_data.shape[0] > 0:
        for i in range(len(unique_ts)):
            # Padding using the last known location of each aircraft
            # Considered aircraft only if data was found within ts_range before current ts
            last_position_data = get_last_ts_data(unique_ts[i], flight_data)
            
            # Temporarily removed time dimension from the array
#             ts_array = unique_ts[i] - subtract_idx
            # Mapping lowest time index to 0 (including history: negative)
            # Mapping highest time index to (time_dim - 1)
#             ts_array = (ts_array < 0) * 0 + (ts_array >= 0) * ts_array
            for j in range(last_position_data.shape[0]):
                x_array = last_position_data['x'].iloc[j]
                y_array = last_position_data['y'].iloc[j]
                z_array = last_position_data['z'].iloc[j]
#                 if ts_array != 0:
#                     print((x_array, y_array, z_array, ts_array))
                output_array.append(x_array, y_array, z_array, 1)
    
    return output_array





# For a given list of timestamps (arranged in reverse order, obtain the space of environment at the time

def get_range_df(rng, flight_data, x_length, y_length, z_length, with_ts_range = True, ts_range = None):
#     space_X = np.zeros((len(rng), x_length, y_length, z_length, time_dim))
    space_X = []
    flight_data_subset = flight_data
    for i in rng:
        space_X.append(create_space_ts(ts = i, flight_data = flight_data_subset, x_length = x_length,
                    y_length = y_length, z_length = z_length, with_ts_range = with_ts_range,
                    ts_range = ts_range))
        flight_data_subset = flight_data_subset[flight_data_subset['ts'] <= i]
    
    return space_X





## Discrete Action

# Note: 'final' refers to the state in immediate future, 'initial' refers to the current state

# Action is defined by
# - sign of difference between final ground speed and initial ground speed
# - sign of difference between final altitude and initial altitude
# - quadrant of final azimuth (absolute)

def get_action_discrete(controlled_ac_df, idx):
    current_row = controlled_ac_df.iloc[idx]
    next_row = controlled_ac_df.iloc[idx+1]
    
    # Coding: 0,1,2,3: heading quadrant - 1
    heading = np.ceil(next_row['azimuth']*2/np.pi)-1
    heading = (heading > 0) * heading + (heading <= 0) * 0
    
    # Coding: 0: deceleration, 1: same speed, 2: acceleration
    d_speed = (np.sign(next_row['speed'] - current_row['speed']) + 1)
    
    # Coding: 0: descent, 1: same altitude, 2: ascent
    d_altitude = (np.sign(next_row['altitude'] - current_row['altitude']) + 1)
    
    # The following code works even though this is not a perfect base 3 system
    # because only the leftmost digit exceeds 2
    return(9*heading + 3*d_speed + d_altitude)





## Continuous Action

# - change in speed
# - change in altitude
# - change in heading

def get_action_continuous(controlled_ac_df, idx):
    current_row = controlled_ac_df.iloc[idx]
    next_row = controlled_ac_df.iloc[idx+1]
    heading_change = next_row['azimuth'] - current_row['azimuth']
    altitude_change = next_row['altitude'] - current_row['altitude']
    speed_change = next_row['speed'] - current_row['speed']
    return np.array([speed_change, altitude_change, heading_change])






# The following function generates the 100 * 100 * 100 environment and action at each time step for a given 'controlled aircraft'

def get_env_action_aircraft(controlled_ac_df, x_length, y_length, z_length, space_X, env = True, discrete_action = True):
    uniq_ts = controlled_ac_df['ts'].unique()
    uniq_ts.sort()
    uniq_ts = uniq_ts[:-1]
    
    # Last row does not have any action as future of last row is not observed
    if env:
        space_ac = [space_X[ts-1] for ts in uniq_ts]
    else:
        space_ac = [Sparse3dArray(shape = (x_length, y_length, z_length), dtype = np.int16) for ts in uniq_ts]
    
    if discrete_action:
        actions = np.zeros(len(uniq_ts))
    else:
        actions = np.zeros((len(uniq_ts), 3))
    # The following step can be done without loop by creating lag variable. It will be fixed later
    for i, ts in enumerate(uniq_ts):
        controlled_ac_ts_dat = controlled_ac_df[controlled_ac_df['ts'] == ts]
        x = controlled_ac_df['x'].iloc[0]
        y = controlled_ac_df['y'].iloc[0]
        z = controlled_ac_df['z'].iloc[0]
        space_ac[i].append(x, y, z, -1)
#         space_ac[i, x, y, z, (time_dim - 1)] -= 1
        if discrete_action:
            actions[i] = get_action_discrete(controlled_ac_df, i)
        else:
            actions[i, :] = get_action_continuous(controlled_ac_df, i)
    
    return [[space_ac], [actions]]




# Clean-up functions for creating example data, splitting into training, testing

def get_X_Y(uniq_id, flight_data, x_length, y_length, z_length, space_X, discrete_action = True):
    X = Y = []
    for i, id in enumerate(uniq_id):
        print(i)
        controlled_ac_df = flight_data[flight_data['id'] == id]
        x, y = get_env_action_aircraft(controlled_ac_df, x_length, y_length, z_length, space_X, env = True, discrete_action = discrete_action)
        X.append(x)
        Y.append(y)
    
    return [X, Y]

def fix_XY(XY):
    X = [x for x in XY[0] if (type(x[0]) != np.ndarray and type(x[0]) != np.array)]
    Y = [y for y in XY[0] if (type(y[0]) == np.ndarray or type(y[0]) == np.array)]
    X = X + [x for x in XY[1] if (type(x[0]) != np.ndarray and type(x[0]) != np.array)]
    Y = Y + [y for y in XY[1] if (type(y[0]) == np.ndarray or type(y[0]) == np.array)]
    return [X, Y]


def unlist(list_of_list):
    out_list = []
    for temp_list in list_of_list:
        out_list = out_list + temp_list
    
    return out_list

def bind(list_of_array):
    return np.concatenate(list_of_array)

def train_test_split(X, Y, split_ratio = 0.7):
    end_index = math.floor(split_ratio * len(X))
    train_X = X[0:end_index]
    train_Y = Y[0:end_index]
    test_X = X[end_index:]
    test_Y = Y[end_index:]
    return [train_X, train_Y, test_X, test_Y]


