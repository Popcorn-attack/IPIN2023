import torch
from torch.utils.data import DataLoader,Dataset
import re
import numpy as np
import  torch.nn as nn
import sys
from dataset import IPINDataset
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)



loss = torch.nn.MSELoss()

def get_net(feature_num):
    net = nn.Linear(feature_num, 2)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()
def get_y(label):
    step0 = label[:,0,0:2]
    step1 = label[:,1,0:2]
    #length = torch.sqrt((step1[0]-step0[0])**2+(step1[1]-step0[1])**2)
    #angle = torch.arctanh(1.0)
    return step1-step0

def train(net, train_data,labels,num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset =  IPINDataset(train_data,labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    net = net.float()
    for epoch in range(num_epochs):
        for X, Y in train_iter:
            y = get_y(Y)
            X = X.flatten(start_dim =1 )
            pred_y = net(X.float())
            l = loss(pred_y, y.float())
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        #train_ls.append(log_rmse(net, X, y))
        train_ls.append(l)
        print(train_ls)
        #if test_labels is not None:
        #    test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

if __name__ == '__main__':
    data = np.load('./dataset/step_100_rawdata.npy',allow_pickle=True)
    data = np.array(data)
    data=np.vstack(data).astype(np.float)

    label = np.load('./dataset/step_100_label.npy',allow_pickle=True)
    label = np.array(label)
    label=np.vstack(label).astype(np.float)

    net = get_net(900)

    train_log,_ = train(net,data,label,100,5,0,64)


