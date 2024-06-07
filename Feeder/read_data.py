import os
import random
import numpy as np
from PIL import Image
from ..Checkpoint import parameters
import encode_label

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

def read_dataset(path, status,seed=0,balance=True):
    all_label_list = []
    Folder_name = ['%03d'%(i) for i in range(1,parameters.NUM_CLASSESS+1)]
    num_classess = parameters.NUM_CLASSESS

    max_len = 0

    for i in range(num_classess):
        dir_path = os.path.join(path, status, Folder_name[i])
        label_list_per_class = os.listdir(dir_path)
        label_list_per_class = [os.path.join(dir_path,l) for l in label_list_per_class]

        all_label_list.append(label_list_per_class)
        
        if len(label_list_per_class)> max_len:
            max_len = len(label_list_per_class)
    
    shuffle_all_label_list =[]
    
    # deal with class imbalance by copying samples.
    if  balance == True: 
        for i in range(len(all_label_list)):
            num = int(np.ceil(max_len/len(all_label_list[i])))
            
            z = []
            for j in range(num):
                z = z + all_label_list[i]
            shuffle_all_label_list = shuffle_all_label_list + z[0:max_len]
    else:
        for i in range(len(all_label_list)):
            shuffle_all_label_list = shuffle_all_label_list + all_label_list[i]
         
    random.seed(seed)
    random.shuffle(shuffle_all_label_list)

    return shuffle_all_label_list


def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
    return np.array(images)

def read_batch(i,batch_size,all_clips_name):
    start = (i*batch_size) % len(all_clips_name)
    end = min(start+batch_size,len(all_clips_name))
    
    batch_clips_name = all_clips_name[start:end]
    clip_Y = encode_label.onehotencode(batch_clips_name)
    clip_X = np.zeros([(end - start)*2,
                       parameters.NUM_FRAMES,
                       parameters.CROP,
                       parameters.CROP,
                       3], dtype=np.float32)
    j=0
    for i,path in enumerate(batch_clips_name):
        for samplename in sorted(os.listdir(path)):
            image = load_images_from_folder(os.path.join(path,samplename))
            clip_X[j, :, :, :, :] = image
            j+=1
    return clip_Y,clip_X
