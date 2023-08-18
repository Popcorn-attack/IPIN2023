from FYLReadData import read_file
import numpy as np
#import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal 

import torch
from torch.utils.data import DataLoader,Dataset


#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(xyz)
#o3d.visualization.draw_geometries([pcd])
#print(xyz)
 
#raw = read_file('release/trials/0_3_52_pdr.txt')
#pos3 = raw['pos3']
#acce = np.array(raw['acce'])
#gyro = np.array(raw['gyro']) 
#magn = np.array(raw['magn'])

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
    #plt.show()

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
    gt_T =  pos3.loc[:,'%time'].values
    x = pos3.loc[:,'x'].values
    y = pos3.loc[ :,'y'].values
    z = pos3.loc[ :,'z'].values
    p_q = pos3.loc[ :,'x':'q3'].values

    peaks,_ = StepDetection_for_whole_data(acce)
    peaks_time  = acce[peaks][:,0]
    
    batches = []
    label= []
    for i in range(len(peaks_time)-1):
        t0 = peaks_time[i]
        t1 = peaks_time[i+1]
        #step = filter(lambda x:t0 <= x <= t1 , gyro[:,0])
        one_step_acce = np.where((t0 <= acce[:,0])&(gyro[:,0] <= t1))
        one_step_acce = acce[one_step_acce][:,2:5]
        
        one_step_gyro = np.where((t0 <= gyro[:,0])&(gyro[:,0] <= t1))
        one_step_gyro = gyro[one_step_gyro][:,2:5]
        one_step_magn = np.where((t0 <= magn[:,0])&(magn[:,0] <= t1))
        one_step_magn = magn[one_step_magn][:,2:5]
        list_100 = np.zeros([200,9])
        
        norms_acce =np.linalg.norm(one_step_acce,axis =1)
        one_step_acce_normed = np.divide(one_step_acce,norms_acce[:,np.newaxis])
        #print(one_step_acce_normed) 
        norms_gyro =np.linalg.norm(one_step_gyro,axis =1)
        one_step_gyro_normed = np.divide(one_step_gyro,norms_gyro[:,np.newaxis])
        #print(one_step_acce_normed) 
        norms_magn =np.linalg.norm(one_step_magn,axis =1)
        one_step_magn_normed = np.divide(one_step_magn,norms_magn[:,np.newaxis])
        #print(one_step_acce_normed) 
        list_100[0:len(one_step_acce),0:3] = one_step_acce_normed
        list_100[0:len(one_step_gyro),3:6] = one_step_gyro_normed
        list_100[0:len(one_step_magn),6:9] = one_step_magn_normed
        #print(list_100)

        one_step_gt = np.where((t0 <= gt_T)&(gt_T <= t1))
        one_step_gt_pq = [p_q[one_step_gt[0][0],:],p_q[one_step_gt[-1][-1],:]]
        


        
        label.append(one_step_gt_pq)
        #name_list = ['acce','gyro','magn','xyzq']
        #batch = list(zip(name_list,batch))
        batches.append(list_100)

    batches = np.array(batches)
    label = np.array(label)
    batches = [batches,label]
    name = ['batches',';label']
    dataset = list(zip(name,batches))
    np.save(pdr_file,dataset)
    #np.save(pdr_file,batches)
    #print(batches)
    #print(peaks_time)
if __name__=='__main__':
    
    #write_batch('0_0_51_pdr.txt','0_0_gt.csv')     
    data = np.load('0_0_51_pdr.txt.npy',allow_pickle=True)
    print(data)
