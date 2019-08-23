from geopy import distance
import numpy as np
import math
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

def repeat(x, times):
    lis = [x for i in range(times)]
    return lis

def unlist(lis):
    ret = []
    for i in lis:
        ret += i
    return ret

def get_xyz(row, conversion_y = 0.0829, conversion_z = 500):
    conversion_x = 9.26/(math.cos(row[0] * math.pi/180) * 111.699)
    y = int(np.ceil((row[0] - 35)/conversion_y))
    x = int(np.ceil((row[1] + 78)/conversion_x))
    z = int(np.ceil(row[2]/conversion_z))
    return (x, y, z)

def isinrange(pos, lat_len, lon_len, alt_len = 100):
    return pos[0] >= 0 and pos[0] <= 100 and pos[1] >= 0 and pos[1] <= 100 and pos[2] >= 0 and pos[2] <= alt_len

def check_collision(pos, reference):
    return pos[2] == reference[2] and pos[1] == reference[1] and pos[0] == reference[0]

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
