
import os
import cv2
from PIL import Image
from tqdm import tqdm


if not os.path.exists("drone_factory_frames"):
    os.mkdir("drone_factory_frames")

path_to_save = os.path.relpath("drone_factory_frames", ".")


pbar = tqdm(total=400*1)
for k in range(1):
    frame_idx = 1
    cap = cv2.VideoCapture("drone_factory.mp4")

    while(cap.isOpened() and frame_idx<401):

        ret, frame = cap.read()
        if(ret == True):

            name = 'frame_{}_{}.jpg'.format(k, frame_idx)
            cv2.imwrite(os.path.join(path_to_save, name), frame)
            frame_idx += 1
        
        else:
            continue

    pbar.update(400)

cap.release()