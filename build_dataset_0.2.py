from FYLReadData import read_file
import numpy as np
#import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal 
import os
import glob
import torch
from torch.utils.data import DataLoader,Dataset
import re
from peak_detection_using_Acce import StepDetection_for_whole_data
import math
from utility import show_label_path,get_y_length
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

def StepDetection_for_whole_data1(Accedata):
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



def write_batch(pdr_file,gt_file,pdr_name,gt_name):
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
    xy = np.array([x,y]).transpose()

    peaks,_ = StepDetection_for_whole_data(acce)
    peaks_time  = acce[peaks][:,0]
    peak_time_filtered = []
    batches = []
    label= []
    for i in range(len(peaks_time)-1):
        t0 = peaks_time[i]
        t1 = peaks_time[i+1]
        #step = filter(lambda x:t0 <= x torch<= t1 , gyro[:,0])
        if t1-t0 <1.49:
            list_100 = np.zeros([150,9])

            one_step_gt = np.where((t0 <= gt_T)&(gt_T <= t1))
            one_step_gt_pq = [p_q[one_step_gt[0][0],:],p_q[one_step_gt[-1][-1],:]]
            step0 = one_step_gt_pq[0][0:2]
            step1 = one_step_gt_pq[1][0:2]
            length = np.sqrt((step1[0]-step0[0])**2+(step1[1]-step0[1])**2)
            #length = get_y_length(one_step_gt_pq)\
            if length<0.99 and length>0.25 :

                peak_time_filtered.append([t0,t1])
                one_step_acce = np.where((t0 <= acce[:,0])&(gyro[:,0] <= t1))
                one_step_acce = acce[one_step_acce][:,2:5]
                
                one_step_gyro = np.where((t0 <= gyro[:,0])&(gyro[:,0] <= t1))
                one_step_gyro = gyro[one_step_gyro][:,2:5]
                one_step_magn = np.where((t0 <= magn[:,0])&(magn[:,0] <= t1))
                one_step_magn = magn[one_step_magn][:,2:5]
                
                

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
                # .2465252876281738
                # 23.386489391326904 1.3024013042449951
                # 29.299606561660767 1.190260410308838
                # 33.485981941223145 1.1622653007507324
                # 36.81916260719299 1.2518

        
            
                label.append(one_step_gt_pq)
                #name_list = ['acce','gyro','magn','xyzq']
                #batch = list(zip(name_list,batch))
                batches.append(list_100)
            else:
                print(length)
            
        else:
            print(t0,t1-t0)
            pass
    peak_time_filtered = np.array(peak_time_filtered)
    #peak_time_filtered_unique = np.unique(peak_time_filtered)
    # batches = np.array(batches)step_100_rawdata
    # label = np.array(label)
    #batches = [batches,label]
    #name = ['batches',';label']
    #dataset = list(zip(name,batches))
    # print(batches.shape)
    # print(label.shape)
    #np.save(pdr_name,batches)
    #np.save(gt_name,label)
    return batches,label,peak_time_filtered
    #np.save(pdr_file,batches)
    #print(batches)
    #print(peaks_time)


def write_pdr_3(pdr_file,peak_time):
    raw = read_file(pdr_file)
    acce = np.array(raw['acce'])
    gyro = np.array(raw['gyro']) 
    magn = np.array(raw['magn'])

    peaks_time = peak_time
    batches = []
    label= []
    for i in peak_time:
        t0 = i[0]
        t1 = i[1]
        #step = filter(lambda x:t0 <= x torch<= t1 , gyro[:,0])
        if t1-t0 <1.5:
            one_step_acce = np.where((t0 <= acce[:,0])&(gyro[:,0] <= t1))
            one_step_acce = acce[one_step_acce][:,2:5]
            
            one_step_gyro = np.where((t0 <= gyro[:,0])&(gyro[:,0] <= t1))
            one_step_gyro = gyro[one_step_gyro][:,2:5]
            one_step_magn = np.where((t0 <= magn[:,0])&(magn[:,0] <= t1))
            one_step_magn = magn[one_step_magn][:,2:5]
            list_100 = np.zeros([150,9])
            

            acce_max, acce_min = np.max(one_step_acce),np.min(one_step_acce)
            one_step_acce = (one_step_acce-acce_min)/(acce_max-acce_min)
            gyro_max, gyro_min = np.max(one_step_gyro),np.min(one_step_gyro)
            one_step_gyro = (one_step_gyro-gyro_min)/(gyro_max-gyro_min)
            magn_max, magn_min = np.max(one_step_magn),np.min(one_step_magn)
            one_step_magn = (one_step_magn-magn_min)/(magn_max-magn_min)
            





            list_100[0:len(one_step_acce),0:3] = one_step_acce
            list_100[0:len(one_step_gyro),3:6] = one_step_gyro
            list_100[0:len(one_step_magn),6:9] = one_step_magn


            #name_list = ['acce','gyro','magn','xyzq']
            #batch = list(zip(name_list,batch))
            batches.append(list_100)
            
        else:
            print(t0,t1-t0)
            pass
    return batches            




def splitedataset(dataset,label):
    length = dataset.length()
    train_index = np.random.random_integers(0,length,6000)
    train_dataset = dataset[train_index,:,:]
    train_label = label[train_index,:,:]

def return_length(path_label,xy):
    lenth_label = 0
    length_gt = 0
    for label in path_label:
        step0 = label[0][0:2]
        step1 = label[1][0:2]
        length = math.sqrt((step1[0]-step0[0])**2+(step1[1]-step0[1])**2)
        lenth_label+=length

    gt_steps = len(xy)//50 
        
    for i in range(gt_steps-1):
        step0 = xy[i*50]
        step1 = xy[(i+1)*50]
        length_step = math.sqrt((step1[0]-step0[0])**2+(step1[1]-step0[1])**2)
        length_gt+= length_step
    return lenth_label,length_gt

def write_batch_51(pdr_file,gt_file):
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
    xy = np.array([x,y]).transpose()

    # peaks,_ = StepDetection_for_whole_data(acce)
    # peaks_time  = acce[peaks][:,0]
    time_list = []
    batches = []
    label= []
    for i in range(len(gt_T)//20-1):
        # t0 = peaks_time[i]
        # t1 = peaks_time[i+1]
        #step = filter(lambda x:t0 <= x torch<= t1 , gyro[:,0])
        t0 = 0.2*i
        t1 = 0.2*(i+1)
        # if t1-t0 <1.49:
        list_100 = np.zeros([30,9])

        one_step_gt = np.where((t0 <= gt_T)&(gt_T <= t1))
        one_step_gt_pq = [p_q[one_step_gt[0][0],:],p_q[one_step_gt[-1][-1],:]]
        step0 = one_step_gt_pq[0][0:2]
        step1 = one_step_gt_pq[1][0:2]
        # length = np.sqrt((step1[0]-step0[0])**2+(step1[1]-step0[1])**2)
        # #length = get_y_length(one_step_gt_pq)\
        # # if length<0.99 and length>0.25 :

        time_list.append([t0,t1])
        one_step_acce = np.where((t0 <= acce[:,0])&(gyro[:,0] <= t1))
        one_step_acce = acce[one_step_acce][:,2:5]
        
        one_step_gyro = np.where((t0 <= gyro[:,0])&(gyro[:,0] <= t1))
        one_step_gyro = gyro[one_step_gyro][:,2:5]

        one_step_magn = np.where((t0 <= magn[:,0])&(magn[:,0] <= t1))
        one_step_magn = magn[one_step_magn][:,2:5]
        
        

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
        # .2465252876281738
        # 23.386489391326904 1.3024013042449951
        # 29.299606561660767 1.190260410308838
        # 33.485981941223145 1.1622653007507324
        # 36.81916260719299 1.2518


    
        label.append(one_step_gt_pq)
        #name_list = ['acce','gyro','magn','xyzq']
        #batch = list(zip(name_list,batch))
        batches.append(list_100)
    # else:
    #     print(length)
        
    #     # else:
        #     print(t0,t1-t0)
        #     pass
    time_list = np.array(time_list)
    #peak_time_filtered_unique = np.unique(peak_time_filtered)
    # batches = np.array(batches)step_100_rawdata
    # label = np.array(label)
    #batches = [batches,label]
    #name = ['batches',';label']
    #dataset = list(zip(name,batches))
    # print(batches.shape)
    # print(label.shape)
    #np.save(pdr_name,batches)
    #np.save(gt_name,label)
    return batches,label,time_list
    #np.save(pdr_file,batches)
    #print(batches)
    #print(peaks_time)


if __name__=='__main__':
    #path_pdr_data = os.pardir
    #path_gt_data = os.pardir('release\gt')
    #pdr_data_names  = glob.glob(r"C:\Users\Shu\Desktop\IPIN\release\trials\*.txt")
    p2 = glob.glob(r'./release/trials/*_pdr.txt')
    p3 = glob.glob(r'./release/gt/*.csv')
    step_100_rawdata = []
    step_100_label = []


    for gts in p3 :

        trail_list4_data = []
        trail_list4_label = []

        for files in p2:
            pdr_file = files.split('/')
            file_name = pdr_file[3].split('.')[0]
            gt_file = gts.split('/')
            num_pdr = re.findall('\d+', pdr_file[3])
            
            num_gt = re.findall('\d+',gt_file[3])

            if num_gt[0] == '16' or num_gt[0] =='17' or num_gt[0] =='22' or num_gt[0] =='23' or num_gt[0] == '11' or num_gt[0] == '12' :
                
                break
            

            if num_pdr[0:2] == num_gt and num_pdr[2] == '51' :
                pdr_file_name = str('dataset/'+ file_name)
                gt_file_name = str('dataset/' + file_name + '_gt' )
                #try:
                print(num_gt)                       
                path_data,path_label,time_list = write_batch_51(files,gts)
                #show_label_path(path_label)
                # total = path_data
                # for i in range(3):
                #     file_NA_ME = files[:-9]+str(i+2)+'_pdr.txt'
                    
                #     batch = write_pdr_3(file_NA_ME,peak_time)
                #     path_data = np.concatenate((np.array(path_data),np.array(batch)),axis=2)


                if path_data != []:
                    step_100_rawdata = step_100_rawdata+ list(path_data)
                    step_100_label = step_100_label+path_label

                    print(file_name,len(path_data))
                        #print('label_length:{},gt_lenth:{}'.format(return_length()))
                        #step_100_rawdata.append(path_data)
                        #step_100_label.append(path_label)
                        #trail_list4_data.append(path_data)
                        #trail_list4_label.append(path_label)
                # except :
                #     print('error:'+ file_name)
                # pass
        #step_100_rawdata = step_100_rawdata+trail_list4_data
        #step_100_label = step_100_label+trail_list4_label
    #splitedataset(step_100_rawdata,step_100_label)

    np.save('./dataset/0.2*30_51_rawdata',step_100_rawdata)
    np.save('./dataset/0.2*30_51_label',step_100_label)

    #print(p2[0],p3[0])
    #write_batch(p2[0],p3[0])     
    #data = np.load('0_0_51_pdr.txt.npy',allow_pickle=True)
    #print(data)
