import os
import sys
import cv2
import random
from ..Checkpoint import parameters

training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, \
                     38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, \
                     80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
training_cameras = [2, 3]
training_setups = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

num_S=32
num_C=3
num_P=103
num_R=2
num_A=120

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = max(total_frames // parameters.NUM_FRAMES, 1)

    frame_indices = [i * interval for i in range(parameters.NUM_FRAMES)]
    gap = parameters.NUM_FRAMES-len(frame_indices)
    frame_indices.extend(frame_indices[:gap])
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (parameters.RESIZE, parameters.RESIZE))

            x = random.randint(0, parameters.RESIZE - parameters.CROP)
            y = random.randint(0, parameters.RESIZE - parameters.CROP)
            frame_cropped = frame_resized[y:y + parameters.CROP, x:x + parameters.CROP]

            if random.random() < parameters.FLIP_FACTOR:
                frame_cropped = cv2.flip(frame_cropped, 1)

            frame_filename = os.path.join(output_dir, f'frame_{i + 1:04d}.jpg')
            cv2.imwrite(frame_filename, frame_cropped)
        else:
            print(f"Frame {frame_idx} could not be read.")

    cap.release()

def gendata(root_path, benchmark):
    data_path = os.path.join(root_path, 'Raw_Data')
    wrong_samples_path = os.path.join(root_path, 'wrong_video.txt')
    if wrong_samples_path != None:
        with open(wrong_samples_path, 'r') as f:
            wrong_samples = [
                line.split('.')[0] for line in f.readlines()
            ]
    else:
        wrong_samples = []

    for p in range(1, num_P + 1):
        for c in range(1, num_C + 1):
            for r in range(1,num_R+1):
                for a in range(1,num_A+1):
                    for s in range(1, num_S + 1):
                        sequence_id = 'S%03dC%03dP%03dR%03dA%03d' \
                                      % (s, c, p, r, a)
                        if sequence_id in wrong_samples:
                            continue
                        if benchmark == 'X-set':
                            split = 'Train' if s in training_setups else 'Val'
                        elif benchmark == 'X-sub':
                            split = 'Train' if p in training_subjects else 'Val'
                        elif benchmark == 'X-view':
                            split = 'Train' if c in training_cameras else 'Val'
                        else:
                            print('\nWrong benchmark!\n')
                            sys.exit(0)

                        video_file = os.path.join(data_path,
                                                  sequence_id + '_rgb.avi')
                        group_dir = os.path.join(root_path,split,'%03d'%(a))
                        if not os.path.exists(group_dir):
                            os.makedirs(group_dir)
                        group_key = 'S%03dS%03dC%03dP%03dR%03dA%03d' \
                                    % ((s-1)//4,s%2, c, p, r, a)
                        group_dir = os.path.join(group_dir, group_key)
                        if not os.path.exists(group_dir):
                            os.makedirs(group_dir)
                        target_dir = os.path.join(group_dir,sequence_id)
                        if not os.path.exists(group_dir):
                            os.makedirs(group_dir)
                        extract_frames(video_file,target_dir)


def make_dir(root_path):
    for i in range(num_A):
        i = str(i)
        path1 = os.path.join(root_path, 'Train', i)
        path2 = os.path.join(root_path, 'Val', i)

        path_list = [path1, path2]
        for dirs in path_list:
            if not os.path.exists(dirs):
                os.mkdir(dirs)
            else:
                print('\nDirectory already exists, please delete it!\n')
                sys.exit(0)



if __name__ == '__main__':
    benchmark = 'X-set'#'X-set', 'X-sub', 'X-view'
    root_path = './NTU120'

    for part in ['Train', 'Val']:
        target_path = os.path.join(root_path, part)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
    gendata(
        root_path,
        benchmark)




