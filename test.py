import os
import sys
from Checkpoint import parameters
import numpy as np
from Feeder import encode_label

from Feeder import read_data as rd
from Inference import save_inference_model


def read_test_data(path=parameters.path,dataset=parameters.DATASET_NAME):
    data_path = os.path.join(path, 'Data', dataset)
    all_clips_name = rd.read_dataset(data_path, 'Test',seed=66,balance=parameters.balance)

    num_clips = int(sys.argv[1])
    clip_Y,clip_X  = rd.read_batch(0,num_clips,all_clips_name)
    return clip_Y,clip_X

def load_clip_name(status,path=parameters.path,dataset=parameters.DATASET_NAME):
    data_path = os.path.join(path,'Data', dataset)
    all_clips_name = rd.read_dataset(data_path, status,seed=66,balance=parameters.balance)
    return all_clips_name


def predict(clip_Y,clip_X):
    acc_count = 0
    prediction = []
    for i in range(clip_X.shape[0]):
        one_clip = clip_X[i,:,:,:,:]
        output = save_inference_model.inference(parameters.PB_MODEL_SAVE_PATH, one_clip)
        if np.argmax(output) == np.argmax(clip_Y[i]):
            acc_count += 1

        pred_name = encode_label.onehotdecode(output[0])
        true_name = encode_label.onehotdecode(clip_Y[i])
        prediction.append({'Output':list(output),'Predicted_class_name':pred_name,'True_calss_name':true_name})
    
    accuracy = (acc_count/(1.0*clip_X.shape[0]))
    return prediction,accuracy
               

def test_net():
    clip_Y,clip_X = read_test_data(parameters.path)
    prediction,accuracy = predict(clip_Y,clip_X)
    print('Clip_accuracy: %g' % accuracy)
    print(prediction)
    return prediction,accuracy
    
    
def main():
    return test_net()
    
    
if __name__ == '__main__':
    prediction,accuracy = main()    
