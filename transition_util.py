from geopy import distance
import numpy as np, math, pandas as pd, itertools

def get_current_pos(row, ts1):
    if row['ts'] != ts1:
        distnc = (row['ground_speed'] * (ts1 - row['ts'])) * 4.63/9
        d = distance.distance(meters = distnc)
        dest = d.destination(point = (row['lat'], row['lon']), bearing = row['azimuth'])
        alt = row['altitude'] + row['climb_rate'] * (ts1 - row['ts'])/60
        if type(alt) == np.array or type(alt) == np.ndarray:
            alt = alt[0]
        return (dest[0], dest[1], alt)
    else:
        return (row['lat'], row['lon'], row['altitude'])

def convert_pos_to_xyz(pos, conversion_y = 0.0829, conversion_z = 500):
    # Minimum required separation is 5 nm = 9.26 km
    conversion_x = 9.26/(math.cos(pos[0] * math.pi/180) * 111.699)
    x = int(np.ceil((pos[1] + 78)/conversion_x))
    y = int(np.ceil((pos[0] - 35)/conversion_y))
    z = int(np.ceil(pos[2]/conversion_z))
    return (x, y, z)

def get_distance_from_airport(x, airport_coords):
    return (math.sqrt((x['x'] - airport_coords[0])**2 + (x['y'] - airport_coords[1])**2))

def convert_pos_to_xyz_new(pos, x_max, y_max, z_max):
    # Minimum required separation is 5 nm = 9.26 km
    x = int(np.ceil((pos[1] + 78) * x_max/10))
    y = int(np.ceil((pos[0] - 35) * y_max/10))
    z = int(np.ceil(pos[2]/500))
    return (x, y, z)

def repeat(x, times):
    lis = [x for i in range(times)]
    return lis

def unlist(lis):
    ret = []
    for i in lis:
        ret += i
    return ret

# def get_xyz(row, conversion_y = 0.0829, conversion_z = 500):
#     conversion_x = 9.26/(math.cos(row[0] * math.pi/180) * 111.699)
#     y = int(np.ceil((row[0] - 35)/conversion_y))
#     x = int(np.ceil((row[1] + 78)/conversion_x))
#     z = int(np.ceil(row[2]/conversion_z))
#     return (x, y, z)

def isinrange(pos, lat_len, lon_len, alt_len = 100):
    return pos[0] >= 0 and pos[0] <= 100 and pos[1] >= 0 and pos[1] <= 100 and pos[2] >= 0 and pos[2] <= alt_len

def check_collision(pos, reference, current_included = True):
    return ((pos == reference).sum() - current_included)

def check_conflict(pos, reference):
    return abs(pos[2] - reference[2]) < 3 and abs(pos[1] - reference[1]) < 3 and abs(pos[0] - reference[0]) < 3

def get_diff_azimuth(azimuth1, azimuth2):
    diff = azimuth1 - azimuth2
    if diff > 180:
        diff = 360 - diff
    elif diff < -180:
        diff = 360 + diff
    return diff

def degrees_to_radians(degrees):
    return math.pi * degrees/180

def radians_to_degrees(radians):
    return radians * 180 / math.pi

def get_nearest_ground_speed_azimuth(ground_speed, azimuth, list_grSpd_azi, two_dim_tuple = False):
    best_match = list_grSpd_azi[0]
#     if two_dim_tuple:
#         best_match = best_match[1]
    azimuth_diff = get_diff_azimuth(azimuth, best_match[1])
    azimuth_diff = degrees_to_radians(azimuth_diff)
    best_match_rate = ground_speed/(best_match[0] * math.cos(azimuth_diff))
    if best_match_rate > 1:
        best_match_rate = 1/best_match_rate
    for i in range(len(list_grSpd_azi)):
        if i != 0:
            check_match = list_grSpd_azi[i]
#             if two_dim_tuple:
#                 check_match = check_match[1]
            azimuth_diff = get_diff_azimuth(azimuth, check_match[1])
            azimuth_diff = degrees_to_radians(azimuth_diff)
            match_rate = ground_speed/(check_match[0] * math.cos(azimuth_diff))
            if match_rate > 1:
                match_rate = 1/match_rate
            if match_rate > best_match_rate:
                best_match = check_match
                best_match_rate = match_rate
    
    return best_match

def append_xyz(landing_ac_data, x_max, y_max, z_max):
    landing_ac_data['x'] = ((landing_ac_data['lon'] + 78) * x_max/10).apply(np.ceil) # -78 to -68
    landing_ac_data['y'] = ((landing_ac_data['lat'] - 35) * y_max/10).apply(np.ceil) # 35 to 45
    landing_ac_data['z'] = (landing_ac_data['altitude']/500).apply(np.ceil) # Observed: 0-43000, but limiting to 0-50000
    landing_ac_data['x'] = landing_ac_data['x'].clip(0, x_max).apply(np.int16)
    landing_ac_data['y'] = landing_ac_data['y'].clip(0, y_max).apply(np.int16)
    landing_ac_data['z'] = landing_ac_data['z'].clip(0, z_max).apply(np.int16)
    return landing_ac_data

def append_xyz_series(series, x_max, y_max, z_max):
    series = pd.concat(series.apply(pd.DataFrame).tolist(), axis=1).T
    series.columns = ["lat", "lon", "altitude"]
    series = append_xyz(series, x_max, y_max, z_max)
    return (series)

def get_next_state(df, transitions1_xyz, nearest_ground_speed_azimuth):
    actions = transitions1_xyz[(df['x'], df['y'], df['z'])][nearest_ground_speed_azimuth]
    temp_delta_ts = delta_ts1[(df['x'], df['y'], df['z'])][nearest_ground_speed_azimuth]
    next_xyzs = []
    for key in actions.keys():
        total = sum(list(actions[key].values()))
        next_xyzs.append(repeat(key, total))
    next_xyzs = unlist(next_xyzs)
    return actions, temp_delta_ts, next_xyzs

def get_next_xyzs(x, y, z, transitions1_xyz):
    return (list(transitions1_xyz[(x, y, z)].keys()))


def get_next_state(df, transitions1_xyz, nearest_ground_speed_azimuth, delta_ts1):
    actions = transitions1_xyz[(df['x'], df['y'], df['z'])][nearest_ground_speed_azimuth]
    temp_delta_ts = delta_ts1[(df['x'], df['y'], df['z'])][nearest_ground_speed_azimuth]
    next_xyzs = []
    for key in actions.keys():
        total = sum(list(actions[key].values()))
        next_xyzs.append(repeat(key, total))
    next_xyzs = unlist(next_xyzs)
    return actions, temp_delta_ts, next_xyzs

def get_greedy_action(next_coords, airport_coords):
    next_coords['dist_from_airport'] = next_coords.apply(lambda x: get_distance_from_airport(x, airport_coords), axis = 1)
    next_coords = next_coords.sort_values(['dist_from_airport']).drop(['dist_from_airport'], axis = 1).reset_index(drop = True)
    return (next_coords.iloc[0].apply(np.int32))

def greedy_update_df(df, airport_coords, transitions1_xyz, delta_ts1, ts1):
    while df['ts'] < ts1:
        nearest_ground_speed_azimuth = get_nearest_ground_speed_azimuth(df['ground_speed'], df['azimuth'], list_grSpd_azi = df['list_grSpd_azi'])
        actions, temp_delta_ts, next_xyzs = get_next_state(df = df, transitions1_xyz = transitions1_xyz, nearest_ground_speed_azimuth = nearest_ground_speed_azimuth, delta_ts1 = delta_ts1)
        actions = pd.concat(pd.Series(list(actions.keys())).apply(pd.DataFrame).tolist(), axis = 1).T
        actions.columns = ['x', 'y', 'z']
        next_xyz = tuple(get_greedy_action(actions, airport_coords))
        next_gr_spd_azi = delta_ts1[(df['x'], df['y'], df['z'])][nearest_ground_speed_azimuth][next_xyz].keys()
        next_ts = delta_ts1[(df['x'], df['y'], df['z'])][nearest_ground_speed_azimuth][next_xyz]
        delta_times = list(next_ts.values())
        delta_times = list(itertools.chain(*delta_times))
        delta_ts = int(min(delta_times))
        gr_spd_azis = [[keys] * len(next_ts[keys]) for keys in next_ts.keys()]
        gr_spd_azi = list(itertools.chain(*gr_spd_azis))
        gr_spd_azi = gr_spd_azi[delta_times.index(delta_ts)]
        df['ts'] = df['ts'] + delta_ts
        df['ground_speed'] = gr_spd_azi[0]
        df['azimuth'] = gr_spd_azi[1]
        df['x'] = next_xyz[0]
        df['y'] = next_xyz[1]
        df['z'] = next_xyz[2]
        if df['z'] != 0:
            df['list_grSpd_azi'] = get_next_xyzs(df['x'], df['y'], df['z'], transitions1_xyz)
        else:
            df['ground_flag'] = True
            break
    return df

def get_transition_values_times(landing_ac_data):
    # Retaining only aircraft IDs with destination == "JFK" in all rows
#     df1 = flight_data[flight_data['jfk_landing_flag']]
#     df1 = df1.sort_values(['id', 'ts']).reset_index(drop = True)
#     df1 = df1.groupby(['id']).apply(lambda x: (x['destination'] == "JFK").mean()).reset_index(drop = False)
#     df1 = df1[['id']][df1[0] == 1.0]
#     landing_ac_data = df1.merge(flight_data)
    landing_ac_data.sort_values(['id', 'ts']).reset_index(drop = True)
    
    transitions1_xyz = {}
    landing_value1_xyz = {}
    time_to_land1_xyz = {}
    delta_ts1 = {}
    condition = True
    previous_id = ''
    landing_ac_data['last_ts'] = landing_ac_data[['id', 'ts']].set_index("id").groupby(level = "id").shift(1).reset_index(drop = True)
    landing_ac_data['last_x'] = landing_ac_data[['id', 'x']].set_index("id").groupby(level = "id").shift(1).reset_index(drop = True)
    landing_ac_data['last_y'] = landing_ac_data[['id', 'y']].set_index("id").groupby(level = "id").shift(1).reset_index(drop = True)
    landing_ac_data['last_z'] = landing_ac_data[['id', 'z']].set_index("id").groupby(level = "id").shift(1).reset_index(drop = True)
    landing_ac_data['last_ground_speed'] = landing_ac_data[['id', 'ground_speed']].set_index("id").groupby(level = "id").shift(1).reset_index(drop = True)
    landing_ac_data['last_azimuth'] = landing_ac_data[['id', 'azimuth']].set_index("id").groupby(level = "id").shift(1).reset_index(drop = True)
    landing_ac_data = landing_ac_data[~landing_ac_data['last_ts'].apply(np.isnan)]
    landing_ac_data = landing_ac_data.sort_values(['id', 'ts']).reset_index(drop = True)
    landing_ac_data = landing_ac_data[(landing_ac_data['last_x'] != landing_ac_data['x']) |
                              (landing_ac_data['last_y'] != landing_ac_data['y']) |
                              (landing_ac_data['last_z'] != landing_ac_data['z'])]
    landing_ac_data = landing_ac_data.sort_values(['id', 'ts']).reset_index(drop = True)
    landing_ac_data['last_x'] = landing_ac_data['last_x'].apply(int)
    landing_ac_data['last_y'] = landing_ac_data['last_y'].apply(int)
    landing_ac_data['last_z'] = landing_ac_data['last_z'].apply(int)
    landing_ac_data['last_ground_speed'] = landing_ac_data['last_ground_speed'].apply(int)
    landing_ac_data['last_azimuth'] = landing_ac_data['last_azimuth'].apply(int)
    
#     unit_time = 5
    start_df = landing_ac_data[['id', 'last_ts', 'last_x', 'last_y', 'last_z']].groupby(['id']).first().reset_index(drop = False)
    start_df.columns = ['id', 'start_ts', 'start_x', 'start_y', 'start_z']
    landing_ac_data = landing_ac_data.merge(start_df)
    landing_ac_data = landing_ac_data.sort_values(['id', 'ts']).reset_index(drop = True)
#     landing_ac_data['t'] = ((landing_ac_data['ts'] - landing_ac_data['start_ts'])/unit_time).apply(np.ceil).apply(int)
#     landing_ac_data['last_t'] = ((landing_ac_data['last_ts'] - landing_ac_data['start_ts'])/unit_time).apply(np.ceil).apply(int)
    
    # Retaining only IDs that have successfully landed at JFK (small range)
    last_df = landing_ac_data.groupby(['id']).last().reset_index(drop = False)
    last_df['coords'] = last_df.apply(lambda x: (x['x'], x['y'], x['z']), axis = 1)
    last_df = last_df[last_df['coords'].apply(lambda x: x[2] <= 3)]
    print(last_df['coords'].value_counts())
    landing_ac_data = landing_ac_data.merge(last_df[['id']])
    landing_ac_data = landing_ac_data.sort_values(['id', 'ts']).reset_index(drop = True)
    landed_xyz = set(last_df['coords'])
    print(landed_xyz)
    
    for i in range(landing_ac_data.shape[0]):
        id1 = landing_ac_data['id'].iloc[i]
        last_x = landing_ac_data['last_x'].iloc[i]
        last_y = landing_ac_data['last_y'].iloc[i]
        last_z = landing_ac_data['last_z'].iloc[i]
        last_ground_speed = landing_ac_data['last_ground_speed'].iloc[i]
        last_azimuth = landing_ac_data['last_azimuth'].iloc[i]
        last_ts = landing_ac_data['last_ts'].iloc[i]
        x = landing_ac_data['x'].iloc[i]
        y = landing_ac_data['y'].iloc[i]
        z = landing_ac_data['z'].iloc[i]
        ground_speed = landing_ac_data['ground_speed'].iloc[i]
        azimuth = landing_ac_data['azimuth'].iloc[i]
        ts = landing_ac_data['ts'].iloc[i]
        landed_ts = landing_ac_data['id_end_ts'].iloc[i]
        condition = (previous_id != id1)
        if (not((last_x, last_y, last_z) in landed_xyz) and condition):
            last_row = True
            try:
                transitions1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] += 1
                landing_value1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] += [R_landing * (gamma ** (landed_ts - ts))]
                time_to_land1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] += [landed_ts - ts]
                delta_ts1[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] += [ts - last_ts]
            except:
                try:
                    transitions1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] = 1
                    landing_value1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] = [R_landing * (gamma ** (landed_ts - ts))]
                    time_to_land1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] = [landed_ts - ts]
                    delta_ts1[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] = [ts - last_ts]
                except:
                    try:
                        transitions1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)] = {(x, y, z): {(ground_speed, azimuth): 1}}
                        landing_value1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)] = {(x, y, z): {(ground_speed, azimuth): R_landing * (gamma ** (landed_ts - ts))}}
                        time_to_land1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)] = {(x, y, z): {(ground_speed, azimuth): landed_ts - ts}}
                        delta_ts1[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)] = {(x, y, z): {(ground_speed, azimuth): [ts - last_ts]}}
                    except:
                        transitions1_xyz[(last_x, last_y, last_z)] = {(last_ground_speed, last_azimuth): {(x, y, z): {(ground_speed, azimuth): 1}}}
                        landing_value1_xyz[(last_x, last_y, last_z)] = {(last_ground_speed, last_azimuth): {(x, y, z): {(ground_speed, azimuth): R_landing * (gamma ** (landed_ts - ts))}}}
                        time_to_land1_xyz[(last_x, last_y, last_z)] = {(last_ground_speed, last_azimuth): {(x, y, z): {(ground_speed, azimuth): landed_ts - ts}}}
                        delta_ts1[(last_x, last_y, last_z)] = {(last_ground_speed, last_azimuth): {(x, y, z): {(ground_speed, azimuth): [ts - last_ts]}}}
        elif last_row:
            previous_id = id1
#             print(previous_id)
            last_row = False
            try:
                transitions1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] += 1
                landing_value1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] += [R_landing * (gamma ** (landed_ts - ts))]
                time_to_land1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] += [landed_ts - ts]
                delta_ts1[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] += [ts - last_ts]
            except:
                try:
                    transitions1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] = 1
                    landing_value1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] = [R_landing * (gamma ** (landed_ts - ts))]
                    time_to_land1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] = [landed_ts - ts]
                    delta_ts1[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)][(x, y, z)][(ground_speed, azimuth)] = [ts - last_ts]
                except:
                    try:
                        transitions1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)] = {(x, y, z): {(ground_speed, azimuth): 1}}
                        landing_value1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)] = {(x, y, z): {(ground_speed, azimuth): R_landing * (gamma ** (landed_ts - ts))}}
                        time_to_land1_xyz[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)] = {(x, y, z): {(ground_speed, azimuth): landed_ts - ts}}
                        delta_ts1[(last_x, last_y, last_z)][(last_ground_speed, last_azimuth)] = {(x, y, z): {(ground_speed, azimuth): [ts - last_ts]}}
                    except:
                        transitions1_xyz[(last_x, last_y, last_z)] = {(last_ground_speed, last_azimuth): {(x, y, z): {(ground_speed, azimuth): 1}}}
                        landing_value1_xyz[(last_x, last_y, last_z)] = {(last_ground_speed, last_azimuth): {(x, y, z): {(ground_speed, azimuth): R_landing * (gamma ** (landed_ts - ts))}}}
                        time_to_land1_xyz[(last_x, last_y, last_z)] = {(last_ground_speed, last_azimuth): {(x, y, z): {(ground_speed, azimuth): landed_ts - ts}}}
                        delta_ts1[(last_x, last_y, last_z)] = {(last_ground_speed, last_azimuth): {(x, y, z): {(ground_speed, azimuth): [ts - last_ts]}}}
    
    pickle.dump(delta_ts1, open("delta_ts1_" + str(x_max) + "_" + str(y_max) + "_" + str(z_max) + "_multi_dict.pkl", "wb"))
    pickle.dump(transitions1_xyz, open("transitions1_xyz_" + str(x_max) + "_" + str(y_max) + "_" + str(z_max) + "_multi_dict.pkl", "wb"))
    pickle.dump(landing_value1_xyz, open("landing_value1_" + str(x_max) + "_" + str(y_max) + "_" + str(z_max) + "_multi_dict.pkl", "wb"))
    pickle.dump(time_to_land1_xyz, open("time_to_land1_" + str(x_max) + "_" + str(y_max) + "_" + str(z_max) + "_multi_dict.pkl", "wb"))
    pickle.dump(landed_xyz, open("landed_xyz.pkl", "wb"))
