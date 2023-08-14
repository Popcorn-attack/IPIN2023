# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:48:37 2021

@author: Yonglei Fan
"""
import  numpy as np
import copy
#对原始数据进行滤波处理，滤波为去掉最大值，去掉最小值，取中间的13个数的均值
def MedianFilter(a):
    a = np.array(a, copy=False, ndmin=1)
    m = copy.deepcopy(a)
    for i in range(11,len(a)-12):
        filter = np.array(a[i-11:i+12])
        filter.sort()
        m[i] = np.average(filter[5:18])
    return m


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False
def read_track_5_data(file_name):
    gyro_list = []
    acce_list = []
    gdrf_list = []
    magn_list = []
    blue_list = []
    with open(file_name) as f:
        for i in f:
            if i[:4] == 'ACCE':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    acce_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'GYRO':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    gyro_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'MAGN':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    magn_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'GDRF':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    gdrf_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'BLUE':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 4:
                    blue_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        total_data_list = [gdrf_list, acce_list, gyro_list, magn_list,blue_list]
        sensor_name_list = ['gdrf', 'acce', 'gyro', 'magn', 'blue']
        return dict(zip(sensor_name_list, total_data_list))




def read_log_file_IPIN2021(file_name):
    posi_list = []
    acce_list = []
    gyro_list = []
    magn_list = []
    pres_list = []
    ligh_list = []
    prox_list = []
    soun_list = []
    ahrs_list = []
    gnss_list = []
    wifi_list = []
    temp_list = []
    humi_list = []
    ble4_list = []
    phone_type = []
    """there are 5 phone models:
        samsung SM-N960F
        samsung SM-N975F
        samsung SM-A520F
        samsung SM-G930F
        Xiaomi Mi 10 Pro
    """
    with open(file_name) as f:
        for i in f:
            if ("SM-N960F" in i):
                phone_type.append("samsung SM-N960F")
            elif ("SM-N975F" in i):
                phone_type.append("samsung SM-N975F")
            elif ("SM-A520F" in i):
                phone_type.append("samsung SM-A520F")
            elif ("SM-G930F" in i):
                phone_type.append("samsung SM-G930F")
            elif ("Xiaomi" in i):
                phone_type.append("Xiaomi Mi 10 Pro")
            elif i[:4] == 'POSI':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    posi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'ACCE':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    acce_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'GYRO':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    gyro_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'MAGN':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    magn_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'PRES':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    pres_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'LIGH':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    ligh_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'PROX':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    prox_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'SOUN':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    soun_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'AHRS':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 10:
                    ahrs_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'GNSS':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 11:
                    gnss_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'WIFI':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    wifi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'TEMP':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    temp_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'HUMI':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    humi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'BLE4':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 9:
                    ble4_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
    #对数据进行平滑处理
    # 0 posi, 1 acce 2 gyro 3 magn 4 pres 5 ligh 6 prox 7 soun 8 ahrs 9 gnss 10 rfid 11 wifi 12 imul 13 imux 14 ble4
    total_data_list = [posi_list, acce_list, gyro_list, magn_list, pres_list, ligh_list, prox_list,
                       soun_list, ahrs_list, gnss_list, wifi_list, temp_list, humi_list, ble4_list,phone_type]
    sensor_name_list = ['posi', 'acce', 'gyro', 'magn', 'pres', 'ligh', 'prox', 'soun', 'ahrs', 'gnss',
                        'wifi', 'temp', 'humi', 'ble4',"phone_type"]
    return dict(zip(sensor_name_list, total_data_list))

#读取数据的时候进行了平滑处理
def read_log_file_IPIN2021Filter(file_name):
    posi_list = []
    acce_list = []
    gyro_list = []
    magn_list = []
    pres_list = []
    ligh_list = []
    prox_list = []
    soun_list = []
    ahrs_list = []
    gnss_list = []
    wifi_list = []
    temp_list = []
    humi_list = []
    ble4_list = []
    phone_type = []
    """there are 5 phone models:
        samsung SM-N960F
        samsung SM-N975F
        samsung SM-A520F
        samsung SM-G930F
        Xiaomi Mi 10 Pro
    """
    with open(file_name) as f:
        for i in f:
            if ("SM-N960F" in i):
                phone_type.append("samsung SM-N960F")
            elif ("SM-N975F" in i):
                phone_type.append("samsung SM-N975F")
            elif ("SM-A520F" in i):
                phone_type.append("samsung SM-A520F")
            elif ("SM-G930F" in i):
                phone_type.append("samsung SM-G930F")
            elif ("Xiaomi" in i):
                phone_type.append("Xiaomi Mi 10 Pro")
            elif i[:4] == 'POSI':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    posi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'ACCE':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    acce_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'GYRO':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    gyro_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'MAGN':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    magn_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'PRES':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    pres_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'LIGH':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    ligh_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'PROX':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    prox_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'SOUN':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    soun_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'AHRS':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 10:
                    ahrs_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'GNSS':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 11:
                    gnss_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'WIFI':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    wifi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'TEMP':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    temp_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'HUMI':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    humi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'BLE4':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 9:
                    ble4_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
    # 0 posi, 1 acce 2 gyro 3 magn 4 pres 5 ligh 6 prox 7 soun 8 ahrs 9 gnss 10 rfid 11 wifi 12 imul 13 imux 14 ble4
    acce_dataframe = np.array(acce_list)
    gyro_dataframe = np.array(gyro_list)
    magn_dataframe = np.array(magn_list)
    """
    acce_dataframe[:,2] = np.convolve(acce_dataframe[:,2], windows_15, mode="same")
    acce_dataframe[:,3] = np.convolve(acce_dataframe[:,3], windows_15, mode="same")
    acce_dataframe[:,4] = np.convolve(acce_dataframe[:,4], windows_15, mode="same")
    gyro_dataframe[:,2] = np.convolve(gyro_dataframe[:,2], windows_15, mode="same")
    gyro_dataframe[:,3] = np.convolve(gyro_dataframe[:,3], windows_15, mode="same")
    gyro_dataframe[:,4] = np.convolve(gyro_dataframe[:,4], windows_15, mode="same")
    magn_dataframe[:,2] = np.convolve(magn_dataframe[:,2], windows_15, mode="same")
    magn_dataframe[:,3] = np.convolve(magn_dataframe[:,3], windows_15, mode="same")
    magn_dataframe[:,4] = np.convolve(magn_dataframe[:,4], windows_15, mode="same")
    """
    acce_dataframe[:, 2] = MedianFilter(acce_dataframe[:, 2])
    acce_dataframe[:, 3] = MedianFilter(acce_dataframe[:, 3])
    acce_dataframe[:, 4] = MedianFilter(acce_dataframe[:, 4])
    gyro_dataframe[:, 2] = MedianFilter(gyro_dataframe[:, 2])
    gyro_dataframe[:, 3] = MedianFilter(gyro_dataframe[:, 3])
    gyro_dataframe[:, 4] = MedianFilter(gyro_dataframe[:, 4])
    magn_dataframe[:, 2] = MedianFilter(magn_dataframe[:, 2])
    magn_dataframe[:, 3] = MedianFilter(magn_dataframe[:, 3])
    magn_dataframe[:, 4] = MedianFilter(magn_dataframe[:, 4])

    acce_list = list(acce_dataframe)
    gyro_list = list(gyro_dataframe)
    magn_list = list(magn_dataframe)

    # #pres数据平滑
    # pres_dataframe = np.array(pres_list)
    # pres_dataframe[:, 2] = MedianFilter(pres_dataframe[:, 2])
    # pres_list = list(pres_dataframe)

    # ligh数据平滑
    ligh_dataframe = np.array(ligh_list)
    ligh_dataframe[:, 2] = MedianFilter(ligh_dataframe[:, 2])
    ligh_list = list(ligh_dataframe)

    total_data_list = [posi_list, acce_list, gyro_list, magn_list, pres_list, ligh_list, prox_list,
                       soun_list, ahrs_list, gnss_list, wifi_list, temp_list, humi_list, ble4_list,phone_type]
    sensor_name_list = ['posi', 'acce', 'gyro', 'magn', 'pres', 'ligh', 'prox', 'soun', 'ahrs', 'gnss',
                        'wifi', 'temp', 'humi', 'ble4',"phone_type"]
    return dict(zip(sensor_name_list, total_data_list))

def read_list_IPIN2021Filter(text_data):
    posi_list = []
    acce_list = []
    gyro_list = []
    magn_list = []
    pres_list = []
    ligh_list = []
    prox_list = []
    soun_list = []
    ahrs_list = []
    gnss_list = []
    wifi_list = []
    temp_list = []
    humi_list = []
    ble4_list = []

    list_data = text_data.split('\n')
    for i in list_data:
        if i[:4] == 'POSI':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 7:
                posi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'ACCE':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 7:
                acce_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'GYRO':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 7:
                gyro_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'MAGN':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 7:
                magn_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'PRES':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 5:
                pres_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'LIGH':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 5:
                ligh_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'PROX':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 5:
                prox_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'SOUN':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 5:
                soun_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'AHRS':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 10:
                ahrs_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'GNSS':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 11:
                gnss_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'WIFI':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 7:
                wifi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'TEMP':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 5:
                temp_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'HUMI':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 5:
                humi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
        elif i[:4] == 'BLE4':
            tmp = i.strip('\n').split(';')
            if len(tmp) == 9:
                ble4_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
    # # 0 posi, 1 acce 2 gyro 3 magn 4 pres 5 ligh 6 prox 7 soun 8 ahrs 9 gnss 10 rfid 11 wifi 12 imul 13 imux 14 ble4
    # acce_dataframe = np.array(acce_list)
    # gyro_dataframe = np.array(gyro_list)
    # magn_dataframe = np.array(magn_list)
    # """
    # acce_dataframe[:,2] = np.convolve(acce_dataframe[:,2], windows_15, mode="same")
    # acce_dataframe[:,3] = np.convolve(acce_dataframe[:,3], windows_15, mode="same")
    # acce_dataframe[:,4] = np.convolve(acce_dataframe[:,4], windows_15, mode="same")
    # gyro_dataframe[:,2] = np.convolve(gyro_dataframe[:,2], windows_15, mode="same")
    # gyro_dataframe[:,3] = np.convolve(gyro_dataframe[:,3], windows_15, mode="same")
    # gyro_dataframe[:,4] = np.convolve(gyro_dataframe[:,4], windows_15, mode="same")
    # magn_dataframe[:,2] = np.convolve(magn_dataframe[:,2], windows_15, mode="same")
    # magn_dataframe[:,3] = np.convolve(magn_dataframe[:,3], windows_15, mode="same")
    # magn_dataframe[:,4] = np.convolve(magn_dataframe[:,4], windows_15, mode="same")
    # """
    # acce_dataframe[:, 2] = MedianFilter(acce_dataframe[:, 2])
    # acce_dataframe[:, 3] = MedianFilter(acce_dataframe[:, 3])
    # acce_dataframe[:, 4] = MedianFilter(acce_dataframe[:, 4])
    # gyro_dataframe[:, 2] = MedianFilter(gyro_dataframe[:, 2])
    # gyro_dataframe[:, 3] = MedianFilter(gyro_dataframe[:, 3])
    # gyro_dataframe[:, 4] = MedianFilter(gyro_dataframe[:, 4])
    # magn_dataframe[:, 2] = MedianFilter(magn_dataframe[:, 2])
    # magn_dataframe[:, 3] = MedianFilter(magn_dataframe[:, 3])
    # magn_dataframe[:, 4] = MedianFilter(magn_dataframe[:, 4])
    #
    # acce_list = list(acce_dataframe)
    # gyro_list = list(gyro_dataframe)
    # magn_list = list(magn_dataframe)
    #
    # #pres数据平滑
    # pres_dataframe = np.array(pres_list)
    # pres_dataframe[:, 2] = MedianFilter(pres_dataframe[:, 2])
    # pres_list = list(pres_dataframe)
    #
    # # ligh数据平滑
    # ligh_dataframe = np.array(ligh_list)
    # ligh_dataframe[:, 2] = MedianFilter(ligh_dataframe[:, 2])
    # ligh_list = list(ligh_dataframe)

    total_data_list = [posi_list, acce_list, gyro_list, magn_list, pres_list, ligh_list, prox_list,
                       soun_list, ahrs_list, gnss_list, wifi_list, temp_list, humi_list, ble4_list]
    sensor_name_list = ['posi', 'acce', 'gyro', 'magn', 'pres', 'ligh', 'prox', 'soun', 'ahrs', 'gnss',
                        'wifi', 'temp', 'humi', 'ble4']
    return dict(zip(sensor_name_list, total_data_list))


def read_log_Point5s_Sample(filename):
    posi_list = []
    acce_list = []
    gyro_list = []
    magn_list = []
    pres_list = []
    ligh_list = []
    prox_list = []
    soun_list = []
    ahrs_list = []
    gnss_list = []
    wifi_list = []
    temp_list = []
    humi_list = []
    ble4_list = []
    with open(filename) as f:
        for i in f:
            if i[:4] == 'POSI':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    posi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'ACCE':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    acce_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'GYRO':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    gyro_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'MAGN':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    magn_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'PRES':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    pres_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'LIGH':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    ligh_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'PROX':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    prox_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'SOUN':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    soun_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'AHRS':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 10:
                    ahrs_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'GNSS':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 11:
                    gnss_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'WIFI':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 7:
                    wifi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'TEMP':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    temp_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'HUMI':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 5:
                    humi_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])
            elif i[:4] == 'BLE4':
                tmp = i.strip('\n').split(';')
                if len(tmp) == 9:
                    ble4_list.append([eval(k) if is_number(k) else k for k in tmp[1:]])

    total_data_list = [posi_list, acce_list, gyro_list, magn_list, pres_list, ligh_list, prox_list,
                       soun_list, ahrs_list, gnss_list, wifi_list, temp_list, humi_list, ble4_list]
    sensor_name_list = ['posi', 'acce', 'gyro', 'magn', 'pres', 'ligh', 'prox', 'soun', 'ahrs', 'gnss',
                        'wifi', 'temp', 'humi', 'ble4']
    maxmim_time = acce_list[-1][0]
    all_simple=[]
    for time_i in range(1,int(maxmim_time)*2):
        time_lim = time_i * 0.5
        posi_simple = []
        acce_simple = []
        gyro_simple = []
        magn_simple = []
        pres_simple = []
        ligh_simple = []
        prox_simple = []
        soun_simple = []
        ahrs_simple = []
        gnss_simple = []
        wifi_simple = []
        temp_simple = []
        humi_simple = []
        ble4_simple = []
        for posi_i in posi_list:
            if abs(posi_i[0]-time_lim)<0.25:
                posi_simple.append(posi_i)
            else:
                continue
        for acce_i in acce_list:
            if abs(acce_i[0]-time_lim)<0.25:
                acce_simple.append(acce_i)
            else:
                continue
        for wifi_i in wifi_list:
            if abs(wifi_i[0]-time_lim)<0.25:
                wifi_simple.append(wifi_i)
            else:
                continue
        for magn_i in magn_list:
            if abs(magn_i[0]-time_lim)<0.25:
                magn_simple.append(magn_i)
            else:
                continue
        for gyro_i in gyro_list:
            if abs(gyro_i[0]-time_lim)<0.25:
                gyro_simple.append(gyro_i)
            else:
                continue
        for pres_i in pres_list:
            if abs(pres_i[0]-time_lim)<0.25:
                pres_simple.append(pres_i)
            else:
                continue
        for ligh_i in ligh_list:
            if abs(ligh_i[0]-time_lim)<0.25:
                ligh_simple.append(ligh_i)
            else:
                continue
        for soun_i in soun_list:
            if abs(soun_i[0]-time_lim)<0.25:
                soun_simple.append(soun_i)
            else:
                continue
        for temp_i in temp_list:
            if abs(temp_i[0]-time_lim)<0.25:
                temp_simple.append(temp_i)
            else:
                continue
        for prox_i in prox_list:
            if (prox_i[0]-time_lim)<0.25:
                prox_simple.append(prox_i)
            else:
                continue
        for humi_i in soun_list:
            if abs(humi_i[0]-time_lim)<0.25:
                humi_simple.append(humi_i)
            else:
                continue
        for gnss_i in gnss_list:
            if abs(gnss_i[0]-time_lim)<0.25:
                gnss_simple.append(gnss_i)
            else:
                continue
        for ahrs_i in ahrs_list:
            if abs(ahrs_i[0]-time_lim)<0.25:
                ahrs_simple.append(ahrs_i)
            else:
                continue
        for ble4_i in ble4_list:
            a = ble4_i[0]-time_lim
            if abs(ble4_i[0]-time_lim)<0.25:
                ble4_simple.append(ble4_i)
            else:
                continue
        simple_data_list = [posi_simple, acce_simple, gyro_simple, magn_simple, pres_simple, ligh_simple, prox_simple,
                           soun_simple, ahrs_simple, gnss_simple, wifi_simple, temp_simple, humi_simple, ble4_simple]
        simple_name_list = ['posi', 'acce', 'gyro', 'magn', 'pres', 'ligh', 'prox', 'soun', 'ahrs', 'gnss',
                            'wifi', 'temp', 'humi', 'ble4']

        all_simple.append(dict(zip(simple_name_list, simple_data_list)))

    return all_simple