#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st


# In[2]:


def threshold_SSD(training_data_ss):
    x1, x2, x3, x4, x5 = {}, {}, {}, {},{}
    m1, m2, m3, m4, m5 = {}, {}, {}, {},{}
    n1, n2, n3, n4, n5 = {}, {}, {}, {},{}
    k1, k2, k3, k4, k5 = {}, {}, {}, {},{}
    s1, s2, s3, s4, s5 = {}, {}, {}, {},{}
    X = [x1, x2, x3, x4, x5]
    max_X = [m1, m2, m3, m4, m5]
    min_X = [n1, n2, n3, n4, n5]
    mean_X =[k1, k2,k3,k4, k5]
    slope_X = [s1,s2,s3,s4, s5] 
    var = ['HWC-VLV','CHWC-VLV','SA-TEMP','SA-SP','SF-SPD']
    training_data_ss = training_data_ss[['HWC-VLV','CHWC-VLV','SA-TEMP','SA-SP','SF-SPD']]
    for v in range(len(var)):
        for i in range(len(training_data_ss)):
            X[v][i] = training_data_ss[var[v]].iloc[i:i+6]
            max_X[v][i] = max(X[v][i])
            min_X[v][i] = min(X[v][i])
            mean_X[v][i] = np.mean(X[v][i])
            if mean_X[v][i] != 0:
                slope_X[v][i] = (max_X[v][i]-min_X[v][i])/mean_X[v][i]
            else:
                slope_X[v][i] = 0
                
    
    slope={}
    for k in range(len(training_data_ss)):
        slope[k] = slope_X[0][k] + slope_X[1][k] + slope_X[2][k] + slope_X[3][k] + slope_X[4][k]
    
    df = pd.DataFrame(slope.items())   
    std_dev = st.stdev(df[1])

    return std_dev*3, df[1]


# In[3]:


def SSD(train_data, Data):
    [std_dev, slopes] = threshold_SSD(train_data)
    [std_dev_, slopes_] = threshold_SSD(Data)
    dff = pd.DataFrame()
    
    for i in range(len(Data)):
        Data2 = pd.DataFrame()
        if  -std_dev <= slopes_[i] <= std_dev:
            Data2 = Data.iloc[i]
            dff = pd.concat([dff, Data2],axis=1)
        
    return dff.T


# In[ ]:





# In[ ]:




