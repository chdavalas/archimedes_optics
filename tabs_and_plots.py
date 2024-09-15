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

smaller_df = stats_df[["test_dataset","distortion_type","method","precision","recall","f1"]]
df_mean = smaller_df.groupby(["test_dataset",  "distortion_type", "method"]).mean().round(3)
df_mean = df_mean.rename(columns={'precision':'precision-mean', 'f1':'f1-mean', 'recall':'recall-mean'})

df_stdv = smaller_df.groupby(["test_dataset",  "distortion_type", "method"]).std().round(3)
df_stdv = df_stdv.rename(columns={'precision':'precision-stdv', 'f1':'f1-stdv', 'recall':'recall-stdv'})

df = pd.concat([df_mean, df_stdv], axis=1)
headers = ['precision-mean','precision-stdv','recall-mean','recall-stdv','f1-mean','f1-stdv']

def combine_columns(row, str1, str2):
    return str(row[str1]) + " \\textpm " + str(row[str2])

# Apply the custom function to create a new column 'Combined'
df['precision'] = df.apply(lambda c : combine_columns(c, 'precision-mean','precision-stdv'), axis=1)
df['recall'] = df.apply(lambda c : combine_columns(c, 'recall-mean','recall-stdv'), axis=1)
df['f1'] = df.apply(lambda c : combine_columns(c, 'f1-mean','f1-stdv'), axis=1)

print(df[["precision", "recall", "f1"]].round(3).to_latex())
fig, axes = plt.subplots(nrows=4, ncols=1,figsize=(10, 8))

plot_df = pd.read_csv('diagnostic_values.csv.stable')

tape = plot_df["window_tape"][0].strip("[]").split(", ")

plot_df = plot_df.rename(columns={'drift_p_val':'p-value', 'driftref':'lower warning limit', 
                                  'mean_image_quality':'mean image quality', 'iqref':'iq lower warning limit',
                                  'lstm_drift_detect':'lstm drift detection', 'lstmref':'lstm upper warning limit',
                                  'class_mean':'classification mean', 'classmeanref':'class mean upper warning limit',
                                  }
                                  )

plot_df[["p-value", "lower warning limit"]].plot(ax=axes[0], yticks=[0.05, 1.0], style=["-", "-."], color=['#1f77b4', 'red'])
axes[0].axvline(x=int(tape[0]), ymin=0, ymax=1, color="gray", linestyle="-.")
axes[0].axvline(x=int(tape[1])-1, ymin=0, ymax=1, color="gray", linestyle="-.")
axes[0].fill_betweenx([0,1], int(tape[0]), int(tape[1])-1, color="gray", alpha=0.2)
axes[0].fill_betweenx([0.0, 0.05], 0, len(plot_df)-1, color="red", alpha=0.2 )
axes[0].set_ylabel("mmd drift")

plot_df = plot_df.rename(columns={'mean_image_quality':'mean image quality', 'iqref':'lower warning limit'})
plot_df[["mean image quality", "iq lower warning limit"]].plot(ax=axes[1], yticks=[1.0, plot_df['iq lower warning limit'][0], 0.0], style=["-", "-."], color=['#1f77b4', 'red'])
axes[1].axvline(x=int(tape[0]), ymin=0, ymax=1, color="gray", linestyle="-.")
axes[1].axvline(x=int(tape[1])-1, ymin=0, ymax=1, color="gray", linestyle="-.")
axes[1].fill_betweenx([0,1], int(tape[0]), int(tape[1])-1, color="gray", alpha=0.2)
axes[1].fill_betweenx([0.0, plot_df['iq lower warning limit'][0]], 0, len(plot_df)-1, color="red", alpha=0.2 )
axes[1].set_ylabel("arniqa mean")


plot_df[["lstm drift detection", "lstm upper warning limit"]].plot(ax=axes[2], yticks=[1.0, 0.5, 0.0], style=["-", "-."], color=['#1f77b4', 'red'])
axes[2].axvline(x=int(tape[0]), ymin=0, ymax=1, color="gray", linestyle="-.")
axes[2].axvline(x=int(tape[1])-1, ymin=0, ymax=1, color="gray", linestyle="-.")
axes[2].fill_betweenx([0,1], int(tape[0]), int(tape[1])-1, color="gray", alpha=0.2)
axes[2].fill_betweenx([0.5, 1.0], 0, len(plot_df)-1, color="red", alpha=0.2 )
axes[2].set_ylabel("lstm drift")


plot_df[["classification mean", "class mean upper warning limit"]].plot(ax=axes[3], yticks=[1.0, 0.5, 0.0], style=["-", "-."], color=['#1f77b4', 'red'])
axes[3].axvline(x=int(tape[0]), ymin=0, ymax=1, color="gray", linestyle="-.")
axes[3].axvline(x=int(tape[1])-1, ymin=0, ymax=1, color="gray", linestyle="-.")
axes[3].fill_betweenx([0,1], int(tape[0]), int(tape[1])-1, color="gray", alpha=0.2)
axes[3].fill_betweenx([0.5, 1.0], 0, len(plot_df)-1, color="red", alpha=0.2 )
axes[3].set_ylabel("class mean")

axes[3].set_xlabel("# of 24-frame instances")
plt.savefig("zurich_blackout_44.jpg")


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