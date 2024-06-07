import os
import sys
import shutil
from Checkpoint import parameters
import numpy as np
from Feeder import read_data as rd
import tensorflow as tf
from Inference import save_inference_model
from Model import VAR


def load_clip_name(status,path=parameters.path,dataset=parameters.DATASET_NAME):
    data_path = os.path.join(path,'Data', dataset)
    all_clips_name = rd.read_dataset(data_path, status,seed=66,balance=parameters.balance)
    return all_clips_name
        
 
def net_placeholder(batch_size=None):
    clip_X = tf.placeholder(tf.float32, shape=[parameters.NUM_FRAMES,
                                               parameters.CROP,
                                               parameters.CROP,
                                               3], name='Input')
    clip_Y = tf.placeholder(tf.float32, shape=[batch_size*2,
                                               parameters.NUM_CLASSESS], name='Label')
    isTraining = tf.placeholder(tf.bool,name='Batch_norm')
    return clip_X,clip_Y,isTraining


def net_loss(clip_Y,logits,c_matrices,clip_logits):
    L_cls = tf.keras.losses.CategoricalCrossentropy(clip_Y,logits)

    diff = clip_logits[:, 1:] - clip_logits[:, :-1]
    T_scene = tf.reduce_mean(tf.square(diff))

    mu = 1.0
    Ccamera_sum = tf.add_n(
        [C_camera(c1, c2, mu) for i, c1 in enumerate(c_matrices) for j, c2 in enumerate(c_matrices) if i != j])

    return L_cls + 0.05*T_scene + 0.1 * Ccamera_sum

def C_2D(V1, V2):
    return tf.norm(V1 - V2, ord='fro')

def C_camera(c1, c2, mu):
    return tf.maximum(mu - tf.norm(c1 - c2, ord='fro'), 0)


def L_VAR(losses, Vs):
    L_VAR_loss = tf.zeros_like(losses)
    for i,V_pairs in enumerate(Vs):
        C2D_sum = tf.add_n([C_2D(V1, V2) for V1, V2 in V_pairs])
        L_VAR_loss[i] = losses[i]+0.1 * C2D_sum

    return tf.reduce_mean(L_VAR_loss, name='loss')


def val(sess,clip_X,clip_Y,isTraining,Softmax_output,test_batch_size,status):
    all_clips_name = load_clip_name(status)
    acc_count = 0
    
    for j in range(len(all_clips_name)):
        if (j*test_batch_size)>len(all_clips_name):
            break
        Y,X = rd.read_batch(j,test_batch_size,all_clips_name)
        feed_dict = {clip_X:X,clip_Y:Y,isTraining:False}
        softmax = sess.run(Softmax_output,feed_dict=feed_dict)

        # Compute clip-level accuracy
        for one_output,one_clip_Y in zip(softmax,Y):
            if np.argmax(one_output) == np.argmax(one_clip_Y):
                acc_count += 1
    accuracy = (acc_count/(len(all_clips_name)*1.0))
    return accuracy


def main():
    all_clips_name = load_clip_name('Train')
    clip_X,clip_Y,isTraining = net_placeholder(None)
    logits,Softmax_output,V_pairs,c_matrices,clip_logits = VAR.var(clip_X,isTraining)
    loss = net_loss(clip_Y,logits,c_matrices,clip_logits)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):        
        train_step = tf.train.AdamOptimizer(parameters.LEARNING_RATE).minimize(loss)
     
    Saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        
        writer = tf.summary.FileWriter(os.path.join(parameters.path, 'Model'), sess.graph)
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)
        
        for i in range(parameters.TRAIN_STEPS):
            Y, X = rd.read_batch(i, parameters.BATCH_SIZE, all_clips_name)
            losses=[]
            Vs=[]
            for i in range(parameters.BATCH_SIZE*2):
                one_clip = X[i, :, :, :, :]
                feed_dict = {clip_X: one_clip, clip_Y: Y, isTraining: True}
                _,loss_,V_pairs = sess.run([train_step,loss,V_pairs],feed_dict=feed_dict)
                losses.append(loss_)
                Vs.append(V_pairs)
            Loss = L_VAR(losses,Vs)

            if i % 100 == 0: 
                # accuracy on val clips
                val_acc = val(sess, clip_X, clip_Y, isTraining, Softmax_output, 1, 'Val')
                print('\nVal_accuracy = %g\n' % (val_acc))
                
                # Way 1 : saving checkpoint model
                if sys.argv[1] == 'CHECKPOINT':                    
                    if os.path.exists(parameters.CHECKPOINT_MODEL_SAVE_PATH) and (i == 0):
                        shutil.rmtree(parameters.CHECKPOINT_MODEL_SAVE_PATH)
                    Saver.save(sess, os.path.join(parameters.CHECKPOINT_MODEL_SAVE_PATH,
                                                  parameters.MODEL_NAME + str(i)))
                        
                #  Way 2 : saving pb model       
                elif sys.argv[1] == 'PB':  
                    if os.path.exists(parameters.PB_MODEL_SAVE_PATH):
                        shutil.rmtree(parameters.PB_MODEL_SAVE_PATH)
                    save_inference_model.save_model(sess, parameters.PB_MODEL_SAVE_PATH, clip_X, Softmax_output, isTraining)
                else:
                    print('The argument is incorrect for the way saving model!')
                    sys.exit(0)
            print('===>Step %d: loss = %g ' % (i,Loss))
     
if __name__ == '__main__':
    main()
