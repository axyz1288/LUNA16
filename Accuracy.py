#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def eval(output, label, threshold):
    TP = 0.
    FP = 0.
    TN = 0.
    FN = 0.
    
    prediction = (output > threshold) * 1
    
    # AN = Actual Negative, AP = Actual Positive
    AN_idx = np.argwhere(label==0)
    AP_idx = np.argwhere(label==1)
    
    # Actual Negative
    if(AN_idx.shape[1] != 0):
        sample_AN_idx = np.random.randint(AN_idx.shape[1], size=5)
        AN_idx = np.dstack((AN_idx[0], AN_idx[1], AN_idx[2]))
        for idx in sample_AN_idx:
            i, j, k = AN_idx[0, idx]
            if(prediction[0, i, j, k] == 1):
                TN += 1
            else:
                FP += 1

    # Actual Positive
    if(AP_idx.shape[1] != 0):
        sample_AP_idx = np.random.randint(AP_idx.shape[1], size=5)
        AP_idx = np.dstack((AP_idx[0], AP_idx[1], AP_idx[2]))
        for idx in sample_AP_idx:
            i, j, k = AP_idx[0, idx]
            if(prediction[1, i, j, k] == 1):
                TP += 1
            else:
                FN += 1
                
    return TP, FP, TN, FN  