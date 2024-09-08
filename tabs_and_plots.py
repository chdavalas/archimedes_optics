import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

stats_df = pd.read_csv('stats.csv.stable')
stats_df["test_dataset"] = stats_df["test_dataset"].astype('string')
stats_df["test_dataset"] = stats_df["test_dataset"].str.strip("[]")
stats_df["test_dataset"] = stats_df["test_dataset"].str.replace(", ", "_")
stats_df["test_dataset"] = stats_df["test_dataset"].str.replace("\'", "")

stats_df["ref_dataset"] = stats_df["ref_dataset"].astype('string')
stats_df["ref_dataset"] = stats_df["ref_dataset"].str.strip("[]")
stats_df["ref_dataset"] = stats_df["ref_dataset"].str.replace(", ", "_")
stats_df["ref_dataset"] = stats_df["ref_dataset"].str.replace("\'", "")

stats_df["precision"] = stats_df["precision"].round(3)
stats_df["recall"] = stats_df["recall"].round(3)
stats_df["f1"] = stats_df["f1"].round(3)

smaller_df = stats_df[["test_dataset","method","distortion_type","precision","recall","f1"]]
df_mean = smaller_df.groupby(["test_dataset", "method", "distortion_type"]).mean().round(3)
df_mean = df_mean.rename(columns={'precision':'precision_mean', 'f1':'f1_mean', 'recall':'recall_mean'})

df_stdv = smaller_df.groupby(["test_dataset", "method", "distortion_type"]).std().round(3)
df_stdv = df_stdv.rename(columns={'precision':'precision_stdv', 'f1':'f1_stdv', 'recall':'recall_stdv'})

df = pd.concat([df_mean, df_stdv], axis=1)
headers = ['precision_mean','precision_stdv','recall_mean','recall_stdv','f1_mean','f1_stdv']
df = df[headers]
print(tabulate(df,headers=headers, tablefmt="latex_raw"))

fig, axes = plt.subplots(nrows=2, ncols=1)

plot_df = pd.read_csv('diagnostic_values.csv.stable')

tape = plot_df["window_tape"][0].strip("[]").split(", ")

plot_df[["drift_p_val", "driftref"]].plot(ax=axes[0], yticks=[1.0, 0.05, 0.0], style=["-", "-."], color=['#1f77b4', 'red'])
axes[0].axvline(x=int(tape[0]), ymin=0, ymax=1, color="gray", linestyle="-.")
axes[0].axvline(x=int(tape[1])-1, ymin=0, ymax=1, color="gray", linestyle="-.")
axes[0].fill_betweenx([0,1], int(tape[0]), int(tape[1])-1, color="gray", alpha=0.2)
plot_df[["mean_image_quality", "iqref"]].plot(ax=axes[1], yticks=[1.0, plot_df["iqref"][0], 0.0], style=["-", "-."], color=['#1f77b4', 'red'])
axes[1].axvline(x=int(tape[0]), ymin=0, ymax=1, color="gray", linestyle="-.")
axes[1].axvline(x=int(tape[1])-1, ymin=0, ymax=1, color="gray", linestyle="-.")
axes[1].fill_betweenx([0,1], int(tape[0]), int(tape[1])-1, color="gray", alpha=0.2)
plt.show()


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