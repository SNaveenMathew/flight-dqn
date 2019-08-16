import pandas as pd
import math
import numpy as np
import sys
sys.path.append('../gym-flight/')
import gym_flight
from gym_flight.utils.geo import distance2d
import multiprocessing as mp
from functools import partial
splits = mp.cpu_count() - 1


# Remove all rows of aircraft with very few rows or very little time in sector
def subset_by_minRow_minTime(flight_data, minRow = 20, minTime_s = 1800):
    max_ts = flight_data.groupby(['id']).raw_ts.max()
    min_ts = flight_data.groupby(['id']).raw_ts.min()
    max_ts = max_ts.reset_index(drop = False)
    min_ts = min_ts.reset_index(drop = False)
    max_ts['raw_ts'] = max_ts['raw_ts'] - min_ts['raw_ts']
    max_ts['raw_ts'] = max_ts['raw_ts'].apply(lambda x: x.delta)/1000000000
    max_ts.columns = ['id', 'total_time_s']
    max_ts = max_ts[max_ts['total_time_s'] >= minTime_s]
    rows_df = flight_data[['id', 'ts']].groupby(['id']).count().reset_index(drop = False)
    rows_df.columns = ['id', 'rows']
    rows_df = rows_df[rows_df['rows'] >= minRow]
    max_ts = pd.merge(max_ts, rows_df, on = ['id'])
    flight_data = pd.merge(flight_data, max_ts, on = ['id'])
    return flight_data


# Remove all rows of aircraft that exceeds maximum time limit in sector
def subset_by_maxTime(flight_data, maxTime_s = 10800):
    max_ts = flight_data.groupby(['id']).raw_ts.max()
    min_ts = flight_data.groupby(['id']).raw_ts.min()
    max_ts = max_ts.reset_index(drop = False)
    min_ts = min_ts.reset_index(drop = False)
    max_ts['raw_ts'] = max_ts['raw_ts'] - min_ts['raw_ts']
    max_ts['raw_ts'] = max_ts['raw_ts'].apply(lambda x: x.delta)/1000000000
    max_ts.columns = ['id', 'total_time_s']
    max_ts = max_ts[max_ts['total_time_s'] <= maxTime_s]
    flight_data = pd.merge(flight_data, max_ts, on = ['id'])
    return flight_data



# Remove all rows of aircraft that exceed a maximum ground speed limit or maximum altitude
def subset_by_maxSpeed_maxAlt(flight_data, maxSpeed = 500, maxAlt = 50000):
    max_speed = flight_data.groupby(['id']).ground_speed.max().reset_index(drop = False)
    max_speed.columns = ['id', 'max_ground_speed']
    print(max_speed.columns)
    max_alt = flight_data.groupby(['id']).altitude.max().reset_index(drop = False)
    max_alt.columns = ['id', 'max_altitude']
    max_speed = pd.merge(max_speed, max_alt, on = ['id'])
    print(max_speed.columns)
    max_speed = max_speed[(max_speed['max_ground_speed'] <= maxSpeed) & (max_speed['max_altitude'] <= maxAlt)]
    flight_data = pd.merge(flight_data, max_speed)
    return flight_data


# Quadrant: azimuth is in 0 to 2*pi. Quadrant = ceiling(azimuth * 2/pi) is in {1, 2, 3, 4}: 0 is unlikely
def get_quadrant(flight_data):
    flight_data['quadrant'] = (flight_data['azimuth'] * 2 / math.pi).apply(math.ceil)
    flight_data['prev_quadrant'] = flight_data[['id', 'quadrant']].set_index(['id']).shift(1).reset_index(drop = True)
    flight_data['quadrant_change'] = (flight_data['quadrant'] != flight_data['prev_quadrant'])
    return flight_data



# Landing aircraft: aircraft altitude <= landed_altitude and is in vicinity of airport and data unavailable for 30+ mins
def get_landed(flight_data, maxTime_s = 1800, landed_altitude = 1000, lower_lon = -74.375, upper_lon = -73.125, lower_lat = 40, upper_lat = 41.25):
    flight_data['landed'] = (flight_data['prev_time_diff_s'] >= maxTime_s) & (flight_data['lon'] >= lower_lon) & (flight_data['lon'] <= upper_lon) & (flight_data['lat'] >= lower_lat) & (flight_data['lat'] <= upper_lat) & (flight_data["altitude"] <= landed_altitude)
    return flight_data


# Entering aircraft: aircraft is currently at the edge or was previous at the edge and data unavailable for 30+ mins
def get_entered(flight_data, maxTime_s = 1800, lower_lon = -77.8, upper_lon = -68.2, lower_lat = 35.2, upper_lat = 44.8):
    prev_lat_lon = flight_data[['id', 'lat', 'lon']].set_index(['id']).shift(1).reset_index(drop = False)
    prev_lat_lon.columns = ['id', 'prev_lat', 'prev_lon']
    flight_data['prev_lat'] = prev_lat_lon['prev_lat']
    flight_data['prev_lon'] = prev_lat_lon['prev_lon']
    flight_data = get_quadrant(flight_data)
    flight_data['entered'] = (((flight_data['lon'] <= lower_lon) | (flight_data['lon'] >= upper_lat) | (flight_data['lat'] <= lower_lat) | (flight_data['lat'] >= upper_lat) | (flight_data['prev_lon'] <= lower_lon) | (flight_data['prev_lon'] >= upper_lat) | (flight_data['prev_lat'] <= lower_lat) | (flight_data['prev_lat'] >= upper_lat)) | flight_data['quadrant_change']) & (flight_data['prev_time_diff_s'] >= maxTime_s)
    return flight_data



# Create new ID for aircraft with data gap >= 1 hour
def get_max_time_diff(flight_data, maxTime_s = 3600):
    flight_data['expired'] = (flight_data['prev_time_diff_s'] >= maxTime_s)
    return flight_data



# Append estimated aircraft number with the id to create new ID and remove aircraft with less than 20 rows or 30 mins data
def get_new_id_flight_data(flight_data, cycle_time = 1800):
    flight_data = get_landed(flight_data)
    flight_data = get_entered(flight_data)
    flight_data = get_max_time_diff(flight_data)
    flight_data['entered_or_landed_or_expired'] = flight_data['entered'] | flight_data['landed'] | flight_data['expired']
    flight_data = flight_data.sort_values(by = ['id', 'ts']).reset_index(drop = True)
    flight_data = flight_data.drop_duplicates(subset = ['id', 'ts'], keep = 'first')
    flight_data['num_id'] = flight_data[['id', 'entered_or_landed_or_expired']].groupby(['id']).entered_or_landed_or_expired.cumsum()
    flight_data['old_id'] = flight_data['id']
    flight_data['id'] = flight_data['id'] + "_" + flight_data['num_id'].astype(int).astype(str)
    flight_data = subset_by_minRow_minTime(flight_data, minRow = 20, minTime_s = 1800)
    flight_data = subset_by_maxTime(flight_data)
    return flight_data


def get_distance2d(lat1, lon1, lat2, lon2):
    if not(np.isnan(lat2)):
        return 1000 * distance2d(lat1, lon1, lat2, lon2)
    else:
        return 0


# Gets the estimated ground speed based on current lat, lon and next lat, lon
def get_estimated_speed(flight_data):
    flight_data['est_ground_speed'] = flight_data[['lat', 'lon', 'next_lat', 'next_lon']].apply(lambda row: get_distance2d(row['lat'], row['lon'], row['next_lat'], row['next_lon'])/60, axis = 1)
    flight_data['est_ground_speed'] = flight_data['est_ground_speed'] * 9 / 4.63
    return flight_data



# Anomalous aircraft: speed at ground > max_speed_ground_altitude (knots) or estimated_speed > max_allowed_ground_speed (knots)
def remove_aircraft_with_anomalous_data(flight_data, speed_at_ground_altitude = True, ground_alt = 1000, max_speed_ground_altitude = 200, estimated_speed = True, max_allowed_est_ground_speed = 500, ground_speed = True, max_allowed_ground_speed = 500, altitude = True, max_allowed_altitude = 50000):
    if speed_at_ground_altitude:
        ground_ac_data = flight_data[flight_data['altitude'] <= ground_alt]
        ground_ac_data = ground_ac_data[['id', 'ground_speed']].groupby(['id']).max().reset_index(drop = False)
    #     ground_ac_data = ground_ac_data[ground_ac_data['ground_speed'] <= max_speed_ground_altitude]
        ground_ac_data.columns = ['id', 'max_speed_ground_altitude']
        flight_data = pd.merge(flight_data, ground_ac_data, how = 'outer', on = 'id')
        speeds = flight_data['max_speed_ground_altitude'].tolist()
        for i in range(len(speeds)):
            if np.isnan(speeds[i]):
                speeds[i] = max_speed_ground_altitude

        flight_data['max_speed_ground_altitude'] = speeds
        flight_data = flight_data[flight_data['max_speed_ground_altitude'] <= max_speed_ground_altitude]
    
    if estimated_speed:
        flight_data = get_estimated_speed(flight_data)
        max_estimated_speed = flight_data[['id', 'est_ground_speed']].groupby(['id']).max().reset_index(drop = False)
        max_estimated_speed.columns = ['id', 'max_est_ground_speed']
        reqd_ac = max_estimated_speed[(max_estimated_speed['max_est_ground_speed'] <= max_allowed_est_ground_speed * 4.63/9)]
        flight_data = pd.merge(flight_data, reqd_ac, on = 'id')
    
    if ground_speed:
        max_ground_speed = flight_data[['id', 'ground_speed']].groupby(['id']).max().reset_index(drop = False)
        max_ground_speed.columns = ['id', 'max_ground_speed']
        max_ground_speed = max_ground_speed[max_ground_speed['max_ground_speed'] <= max_allowed_ground_speed]
        flight_data = pd.merge(flight_data, max_ground_speed, on = 'id')
    
    if altitude:
        max_altitude = flight_data[['id', 'altitude']].groupby(['id']).max().reset_index(drop = False)
        max_altitude.columns = ['id', 'max_altitude']
        max_altitude = max_altitude[max_altitude['max_altitude'] <= max_allowed_altitude]
        flight_data = pd.merge(flight_data, max_altitude, on = 'id')
    
    return flight_data


def fill_time(row, n_min):
    if type(row['next_round_min_ts']) != pd._libs.tslibs.nattype.NaTType:
        return pd.date_range(start = row['round_min_ts'], end = row['next_round_min_ts'], freq = str(n_min) + 'min', closed = 'left')
    else:
        return pd.date_range(start = row['round_min_ts'], end = row['round_min_ts'], freq = str(n_min) + 'min')

def append_id(df1, id_list):
    for i in range(len(df1)):
        temp_df = df1[i]
        temp_df['id'] = id_list[i]
        df1[i] = temp_df
    return df1

def get_timestamps_df(df, n_min):
    df1 = df.apply(lambda row: fill_time(row, n_min), axis = 1).apply(pd.DataFrame).tolist()
    df1 = append_id(df1, df['id'].tolist())
    df1 = pd.concat(df1, axis = 0).reset_index(drop = True)
    return(df1)

def get_timestamps_df_mp(df, n_min = 1):
    p = mp.Pool(processes = splits)
    split_df = np.array_split(df, splits)
    get_timestamps_df_n_min = partial(get_timestamps_df, n_min = n_min)
    pool_results = p.map(get_timestamps_df_n_min, split_df)
    p.close()
    p.join()
    df = pd.concat(pool_results, axis = 0)
    return df