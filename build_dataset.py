from FYLReadData import read_file
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal 

#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(xyz)
#o3d.visualization.draw_geometries([pcd])
#print(xyz)
 
raw = read_file('release/trials/0_3_52_pdr.txt')
#pos3 = raw['pos3']
acce = np.array(raw['acce'])
gyro = np.array(raw['gyro']) 
magn = np.array(raw['magn'])

def show_gt_z(path):
    

    pos3 = pd.read_csv(path)
    xyz = pos3.loc[:,'x':'z'].values

    z = pos3.loc[ :,'z'].values
    a, b =  signal.butter(8,0.1,'low')
    z = signal.filtfilt(a,b,z)
    peaks,_ = signal.find_peaks(z,height = 0,distance = 50)
    #filter = np.where(peaks > 3000)
    #peaks = peaks[filter]
    print(filter)
    print(peaks)
    print(len(z[peaks]))
    plt.plot(z)
    plt.plot(peaks,z[peaks],"x")
    #plt.plot(peaks)
    plt.show()

def StepDetection_for_whole_data(Accedata):
    #首先对数据进行低通滤波处理
    acce_data = np.array(Accedata)
    acce_x, acce_y, acce_z = acce_data[:, 2], acce_data[:, 3], acce_data[:, 4]
    acce_sqrt1 = (np.sqrt((acce_x ** 2 + acce_y ** 2 + acce_z ** 2)))

    b, a = signal.butter(8, 0.1, 'low')
    filtered = signal.filtfilt(b, a, acce_sqrt1)

    #然后对低通滤波后的数据进行波峰波谷检测
    peak_of_filter, _ = signal.find_peaks(filtered, distance=30, prominence=1.5) # 波峰位置
    valley_of_filter, _ = signal.find_peaks(-filtered, distance=30, prominence=1.5) # 波谷位置

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
        if peak_valley[i-1] in peak_of_filter and (peak_valley[i] - peak_valley[i-1]) < 40:
            peak_TRUE.append(peak_valley[i-1])
            valley_TRUE.append(peak_valley[i])
    return peak_TRUE, valley_TRUE



def write_batch(pdr_file,gt_file):
    raw = read_file(pdr_file)
    acce = np.array(raw['acce'])
    gyro = np.array(raw['gyro']) 
    magn = np.array(raw['magn'])

    pos3 = pd.read_csv(gt_file)
    x = pos3.loc[:,'x'].values
    y = pos3.loc[ :,'y'].values
    z = pos3.loc[ :,'z'].values

    peaks,_ = StepDetection_for_whole_data(acce)
    peaks_time  = acce[peaks][:,0]
    
    acce_tem = []
    gyro_tmp = []
    magn_tmp = []
    for i in range(len(peaks_time)):
        t0 = peaks_time[i]
        t1 = peaks_time[i+1]

    print(peaks_time)
write_batch('release/trials/0_0_51_pdr.txt','release/gt/0_0_gt.csv')    
