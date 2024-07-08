import cv2
import os
import csv
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np

#WIP
'''
def generate_video(): 
    image_folder = 'display_frames'
    video_name = 'demo_last_distort.gif'
    
    sorted_frames=[]
    classes = ["normal","blur","noise", "SnP"]
    vid_length = len(os.listdir("display_frames"))

    for idx in range(vid_length):
        i = "frame_{}.jpg".format(idx)
        sorted_frames.append(i)

    drift_p_values = []
    mean_iq_values = []
    steps = []
    with open("iqs.csv", "r") as csvfile:
            reader_variable = csv.reader(csvfile, delimiter=" ")
            for row in reader_variable:
                steps.append(int(row[0]))
                mean_iq_values.append(float(row[1]))


    with open("drifts.csv", "r") as csvfile:
        reader_variable = csv.reader(csvfile, delimiter=" ")
        for row in reader_variable:
            drift_p_values.append(float(row[1]))

    answers = []
    with open("class_choice.csv", "r") as csvfile:
        reader_variable = csv.reader(csvfile, delimiter=",")
        for row in reader_variable:
            answers.append(classes[int(row[0])])

    fig, ax = plt.subplots(3,1)

    ax[1].axhline(0.05, color="r", linestyle="-.")
    ax[1].set_xticks([ k for k in range(0,vid_length+32,32)])
    ax[1].set_ylabel("p-value (below 0.05 indicates drift)")

    ax[2].set_xticks([ k for k in range(0,vid_length+32,32)])
    ax[2].set_ylabel("mean arniqa score ")
    ax[2].set_xlabel("Frame samples")

    plt.ylim(0, 0.7)
    plt.xlim(0,vid_length)

    im_list = []
    
    line1, = ax[1].plot([], [], lw = 3)
    line2, = ax[1].plot([], [], lw = 3)

    for i in range(vid_length):
        im_list.append(plt.imread(os.path.join("display_frames", "frame_{}.jpg".format(i)), "r"))
 

    steps = [0]+steps
    drift_p_values = [drift_p_values[0]]+drift_p_values

    # print(steps)

    def animate(frame_idx):
        ax[0].cla()
        ax[0].set_title("frame:{}, label:{}".format(frame_idx, answers[frame_idx]))
        ax[0].imshow(im_list[frame_idx], animated=True)
        ax[0].set_axis_off()

        if frame_idx==0:
            line1.set_data([steps[0]], [drift_p_values[0]])
            line2.set_data([steps[0]], [mean_iq_values[0]])


        else:
            if frame_idx+1 in steps:
                sli = 1 + (frame_idx+1) // steps[1]
                line1.set_data(steps[:sli], drift_p_values[:sli])
                line2.set_data(steps[:sli], drift_p_values[:sli])


        # print(frame_idx, line.get_data())

        return [line1, line2]
        
    ani = animation.FuncAnimation(
        fig=fig, frames=vid_length, func=animate, interval=85, repeat=False)
    ani.save(video_name, writer = 'ffmpeg')
    plt.show()

generate_video()
'''