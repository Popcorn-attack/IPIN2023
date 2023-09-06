# _*_ coding = utf-8 _*_
'''
date: 2023.07.22
author: Yonglei Fan
主要来进行航向以及步态的估计，和步长的估计
'''
from pdrParameters import pdrParameters
import numpy as np
import scipy.signal as signal
from FYLReadData import *
import matplotlib.pyplot as plt
import os
from geographiclib.geodesic import Geodesic
import math

def StepDetection_for_whole_data(Accedata):
    #首先对数据进行低通滤波处理
    acce_data = np.array(Accedata)
    acce_x, acce_y, acce_z = acce_data[:, 2], acce_data[:, 3], acce_data[:, 4]
    acce_sqrt1 = (np.sqrt((acce_x ** 2 + acce_y ** 2 + acce_z ** 2)))

    b, a = signal.butter(10, 0.06, 'low')
    filtered = signal.filtfilt(b, a, acce_sqrt1)

    peak_of_filter, _ = signal.find_peaks(filtered, distance=35, height=9.90) # 波峰位置
    valley_of_filter, _ = signal.find_peaks(-filtered, distance=35, height=-9.60) # 波谷位置

    plt.figure()
    plt.plot(filtered)
    plt.plot(peak_of_filter, filtered[peak_of_filter], "x")
    plt.show()

    peak_valley = np.concatenate((peak_of_filter, valley_of_filter)) #将波峰波谷位置合并
    peak_valley.sort() # 排序

    '''
    判断step的方法是： 先找到一个波谷，波谷前面必然是一个波峰，且距离应该在0.5Hz*simpleRate范围内(半步之内)
    如果找到一个波谷，距离前面一个波峰太远，则成为该步数为一个无效步数。
    从方法对于实时数据与离线数据都适用
    '''
    valley_TRUE = []
    peak_TRUE = []
    for i in range(len(peak_valley)):
        #先找到一个波谷
        if peak_valley[i] in peak_of_filter:
            continue

        if i == 0:
            continue

        # 此时peak_valley[i]为波谷
        if peak_valley[i-1] in peak_of_filter and (peak_valley[i] - peak_valley[i-1]) < 80:
            peak_TRUE.append(peak_valley[i-1])
            valley_TRUE.append(peak_valley[i])
    return peak_TRUE, valley_TRUE

# 计算两个点的方位角
def calculate_bearing(x1, y1, x2, y2):
    # 计算差值
    delta_x = x2 - x1
    delta_y = y2 - y1

    # 计算方位角（弧度）
    bearing_rad = math.atan2(delta_y, delta_x)

    # 将弧度转换为度数
    bearing_deg = math.degrees(bearing_rad)

    # 调整方位角的范围为[0, 360)度
    if bearing_deg < 0:
        bearing_deg += 360

    return bearing_deg
