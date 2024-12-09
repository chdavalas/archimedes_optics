import matplotlib.pyplot as plt
import numpy as np

issues = ['Blackout', 'Lens Blur', 'Motion Blur']
methods = ['Class-Mean', 'Arniqa-Mean', 'LSTM-Drift', 'MMD-Drift']
colors = ['salmon', 'aquamarine', 'lightblue']

precision = {
    'Blackout': [0.93, 1.0, 0.893, 0.85],
    'Lens Blur': [0.921, 1.0, 0.893, 0.881],
    'Motion Blur': [0.93, 1.0, 0.893, 0.85]
}
recall = {
    'Blackout': [1.0, 0.98, 1.0, 1.0],
    'Lens Blur': [0.797, 0.98, 0.979, 0.898],
    'Motion Blur': [1.0, 1.0, 1.0, 1.0]
}
f1_score = {
    'Blackout': [0.963, 0.99, 0.937, 0.918],
    'Lens Blur': [0.842, 0.99, 0.926, 0.883],
    'Motion Blur': [0.963, 1.0, 0.937, 0.918]
}

# Standard deviations from the table
std_dev_precision = {
    'Blackout': [0.061, 0.0, 0.185, 0.04],
    'Lens Blur': [0.07, 0.0, 0.185, 0.104],
    'Motion Blur': [0.061, 0.0, 0.185, 0.04]
}
std_dev_recall = {
    'Blackout': [0.0, 0.034, 0.0, 0.0],
    'Lens Blur': [0.189, 0.034, 0.036, 0.095],
    'Motion Blur': [0.0, 0.0, 0.0, 0.0]
}
std_dev_f1 = {
    'Blackout': [0.032, 0.017, 0.11, 0.023],
    'Lens Blur': [0.091, 0.017, 0.102, 0.023],
    'Motion Blur': [0.032, 0.0, 0.11, 0.023]
}

x = np.arange(len(methods))  # the label locations
width = 0.2  # the width of the bars

precision_uplim = { i:[0., 0., 0., 0.] for i in ['Blackout', 'Lens Blur', 'Motion Blur']}
for dist in ['Blackout', 'Lens Blur', 'Motion Blur']:
    for i in range(4):
        if precision[dist][i]+std_dev_precision[dist][i]>1:
            precision_uplim[dist][i] = precision[dist][i]+std_dev_precision[dist][i]-1
        else:
            precision_uplim[dist][i] = std_dev_precision[dist][i]

recall_uplim = { i:[0., 0., 0., 0.] for i in ['Blackout', 'Lens Blur', 'Motion Blur']}
for dist in ['Blackout', 'Lens Blur', 'Motion Blur']:
    for i in range(4):
        if recall[dist][i]+std_dev_recall[dist][i]>1:
            recall_uplim[dist][i] = recall[dist][i]+std_dev_recall[dist][i]-1
        else:
            recall_uplim[dist][i] = std_dev_recall[dist][i]

f1_uplim = { i:[0., 0., 0., 0.] for i in ['Blackout', 'Lens Blur', 'Motion Blur']}
for dist in ['Blackout', 'Lens Blur', 'Motion Blur']:
    for i in range(4):
        if f1_score[dist][i]+std_dev_f1[dist][i]>1:
            f1_uplim[dist][i] = f1_score[dist][i]+std_dev_f1[dist][i]-1
        else:
            f1_uplim[dist][i] = std_dev_f1[dist][i]


# Plot with standard deviations
fig, ax = plt.subplots(3, 1, figsize=(12, 12))

for i, issue in enumerate(issues):
    rects1 = ax[i].bar(x - width, precision[issue], width, yerr=(std_dev_precision[issue],precision_uplim[issue]), capsize=5, label='Precision', color=colors[0])
    rects2 = ax[i].bar(x, recall[issue], width, yerr=(std_dev_recall[issue], recall_uplim[issue]), capsize=5, label='Recall', color=colors[1])
    rects3 = ax[i].bar(x + width, f1_score[issue], width, yerr=(std_dev_f1[issue],f1_uplim[issue]), capsize=5, label='F1 Score', color=colors[2])
    
    ax[i].set_ylabel('Scores')
    ax[i].set_title(issue)
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(methods)
    ax[i].legend()

fig.tight_layout()

plt.savefig("factory_inspect.jpg")
#############################################################################################################
precision = {
    'Blackout': [1.0, 0.98, 1.0, 0.944],
    'Lens Blur': [1.0, 0.98, 1.0, 0.944],
    'Motion Blur': [1.0, 0.98, 1.0, 0.944]
}
recall = {
    'Blackout': [1.0, 1.0, 1.0, 1.0],
    'Lens Blur': [1.0, 1.0, 1.0, 1.0],
    'Motion Blur': [1.0, 1.0, 1.0, 1.0]
}
f1_score = {
    'Blackout': [1.0, 0.99, 1.0, 0.971],
    'Lens Blur': [1.0, 0.99, 1.0, 0.971],
    'Motion Blur': [1.0, 0.99, 1.0, 0.971]
}

# Standard deviations from the table
std_dev_precision = {
    'Blackout': [0.0, 0.034, 0.0, 0.056],
    'Lens Blur': [0.0, 0.034, 0.0, 0.056],
    'Motion Blur': [0.0, 0.034, 0.0, 0.056]
}
std_dev_recall = {
    'Blackout': [0.0, 0.0, 0.0, 0.0],
    'Lens Blur': [0.0, 0.0, 0.0, 0.0],
    'Motion Blur': [0.0, 0.0, 0.0, 0.0]
}
std_dev_f1 = {
    'Blackout': [0.0, 0.017, 0.0, 0.03],
    'Lens Blur': [0.0, 0.017, 0.0, 0.03],
    'Motion Blur': [0.0, 0.017, 0.0, 0.03]
}

x = np.arange(len(methods))  # the label locations
width = 0.2  # the width of the bars

# Plot with standard deviations
fig, ax = plt.subplots(3, 1, figsize=(12, 12))

for i, issue in enumerate(issues):
    rects1 = ax[i].bar(x - width, precision[issue], width, yerr=std_dev_precision[issue], capsize=5, label='Precision', color=colors[0])
    rects2 = ax[i].bar(x, recall[issue], width, yerr=std_dev_recall[issue], capsize=5, label='Recall', color=colors[1])
    rects3 = ax[i].bar(x + width, f1_score[issue], width, yerr=std_dev_f1[issue], capsize=5, label='F1 Score', color=colors[2])
    
    ax[i].set_ylabel('Scores')
    ax[i].set_title(issue)
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(methods)
    ax[i].legend()

fig.tight_layout()

plt.savefig("traffic_inspect.jpg")
#############################################################################################################
precision = {
    'Blackout': [1.0, 1.0, 1.0, 0.943],
    'Lens Blur': [1.0, 1.0, 1.0, 0.943],
    'Motion Blur': [1.0, 1.0, 1.0, 0.943]
}
recall = {
    'Blackout': [1.0, 1.0, 1.0, 1.0],
    'Lens Blur': [1.0, 1.0, 0.918, 1.0],
    'Motion Blur': [1.0, 1.0, 1.0, 1.0]
}
f1_score = {
    'Blackout': [1.0, 1.0, 1.0, 0.97],
    'Lens Blur': [1.0, 1.0, 0.956, 0.97],
    'Motion Blur': [1.0, 1.0, 1.0, 0.97]
}

# Standard deviations from the table
std_dev_precision = {
    'Blackout': [0.0, 0.0, 0.0, 0.056],
    'Lens Blur': [0.0, 0.0, 0.0, 0.056],
    'Motion Blur': [0.0, 0.0, 0.0, 0.056]
}
std_dev_recall = {
    'Blackout': [0.0, 0.0, 0.0, 0.0],
    'Lens Blur': [0.0, 0.0, 0.096, 0.0],
    'Motion Blur': [0.0, 0.0, 0.0, 0.0]
}
std_dev_f1 = {
    'Blackout': [0.0, 0.0, 0.0, 0.03],
    'Lens Blur': [0.0, 0.0, 0.053, 0.03],
    'Motion Blur': [0.0, 0.0, 0.0, 0.03]
}
x = np.arange(len(methods))  # the label locations
width = 0.2  # the width of the bars


# Plot with standard deviations
fig, ax = plt.subplots(3, 1, figsize=(12, 12))

for i, issue in enumerate(issues):
    rects1 = ax[i].bar(x - width, precision[issue], width, yerr=std_dev_precision[issue], capsize=5, label='Precision', color=colors[0])
    rects2 = ax[i].bar(x, recall[issue], width, yerr=std_dev_recall[issue], capsize=5, label='Recall', color=colors[1])
    rects3 = ax[i].bar(x + width, f1_score[issue], width, yerr=std_dev_f1[issue], capsize=5, label='F1 Score', color=colors[2])
    
    ax[i].set_ylabel('Scores')
    ax[i].set_title(issue)
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(methods)
    ax[i].legend()

fig.tight_layout()

plt.savefig("zurich_inspect.jpg")