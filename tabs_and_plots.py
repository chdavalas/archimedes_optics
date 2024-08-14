import matplotlib.pyplot as plt
import os

# dir_ = "uav_inspection"
# for i, uav_dir in enumerate(sorted(os.listdir(dir_))):
#     os.rename(dir_+"/"+uav_dir, dir_+"/"+"frame_uav_inspection_{}.jpg".format(i))

# for dataset in 

# plt.subplot(3, 1, 1)
# plt.title(dataset)
# plt.plot(all_drift_p_values)
# plt.axvline(len(test)/2, linestyle="--", color="grey")
# plt.axhline(0.05, linestyle="--", color="red")
# x  = [ i for i, _ in enumerate(all_drift_p_values)]
# plt.fill_between(x, 0, 0.05, alpha=0.3, color="red")
# plt.ylabel("drift")
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.axvline(len(test)/2, linestyle="--", color="grey")
# plt.axhline(0.5, linestyle="--", color="red")
# x  = [ i for i, _ in enumerate(all_drift_p_values)]
# plt.fill_between(x, 0, 0.5, alpha=0.3, color="red")
# plt.plot(mean_iqscore_values, color="purple")
# plt.ylabel("mean image\nquality score")
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.axvline(len(test)/2, linestyle="--", color="grey")
# plt.axhline(0.5, linestyle="--", color="red")
# x  = [ i for i, _ in enumerate(all_drift_p_values)]
# plt.fill_between(x, 0.5, 1, alpha=0.3, color="red")
# plt.plot(all_lstm_mean_values, color="blue")
# plt.ylabel("drift detected\nlstm score")
# plt.xlabel("# of batches")
# plt.grid()

# plt.savefig("current_results.jpg")



# import os

# all_images = os.listdir("interlaken_inspection")
# all_images = sorted([alli for alli in all_images])

# print(all_images); input()
# for i, dir_ in enumerate(all_images):
#     name = "frame_interlaken_inspection_{}.png".format(i)
#     os.rename("interlaken_inspection/"+dir_, "interlaken_inspection/"+name)