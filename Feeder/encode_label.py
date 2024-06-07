import os
import numpy as np
from ..Checkpoint import parameters

path = os.path.dirname(os.getcwd())

def read_labels(data_path):
    samples_categories = []

    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)
        if os.path.isdir(category_path):
            for sample in os.listdir(category_path):
                sample_path = os.path.join(category_path, sample)
            if os.path.isdir(sample_path):
                samples_categories.append(int(category))

    return samples_categories

def onehotcode_all_classses():
    num_classess = parameters.NUM_CLASSESS
    oneHotEncode = np.zeros(shape=[num_classess,num_classess])
    
    oneHotEncodeDict = {}
    Class_name = ['%03d' % (i) for i in range(1, parameters.NUM_CLASSESS + 1)]
    
    for i in range(num_classess):
        oneHotEncode[i][i] = 1.0
        oneHotEncodeDict[Class_name[i]] = oneHotEncode[i][:] 
    return oneHotEncodeDict


def onehotencode(video_name_list):
    label = []
    oneHotEncodeDict = onehotcode_all_classses()
    
    for i in range(len(video_name_list)):
        label_name = video_name_list[i].split('\\')[-2]
        label.append(oneHotEncodeDict[label_name])
    label = np.array(label,dtype=np.float32)
    return label


def onehotdecode(one_hot_code):
    one_hot_code = list(one_hot_code)
    max_value_index = one_hot_code.index(max(one_hot_code))
    oneHotEncodeDict = onehotcode_all_classses()
    
    for class_name,code in oneHotEncodeDict.items():
        max_idx = np.argmax(code)
        if max_value_index==max_idx:
            return class_name
