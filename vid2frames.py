
import os
import cv2
from sys import argv



filename = argv[1]

folder_name = filename.split("/")[-1][:-4]

if not os.path.exists(folder_name):
    os.mkdir(folder_name)

path_to_save = os.path.relpath(folder_name, ".")

cap = cv2.VideoCapture(filename)
frame_idx = 0
while(cap.isOpened()):
    
    ret, frame = cap.read()

    if(ret == True):
        name = 'frame_{}_{}.jpg'.format(folder_name, frame_idx)
        cv2.imwrite(os.path.join(path_to_save, name), frame)
        frame_idx+=1
    else:
        break

cap.release()