import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use("ggplot")

fig = plt.figure()
subplot_1 = fig.add_subplot(2,2,1)
subplot_2 = fig.add_subplot(2,2,2)
subplot_3 = fig.add_subplot(2,1,2)

def animate(i):
    pullData = open("twitter-out.txt","r").read()
    lines = pullData.split('\n')

    negative_count = 0
    positive_count = 0
    neg_confidence_list = []
    pos_confidence_list = []

    all_sentiments_list = []

    for line in lines:
        line = line.split(',')
        if len(line) > 1:

            if line[0] == 'positive':
                positive_count += 1
                pos_confidence_list.append(float(line[1]))
                all_sentiments_list.append(1)
            elif line[0] == 'negative':
                negative_count += 1
                neg_confidence_list.append(float(line[1]))
                all_sentiments_list.append(0)
            else:
                continue

    if len(lines) > 2:
        # pos/neg plot
        bar_heights = [negative_count, positive_count]
        subplot_1.clear()
        subplot_1.bar([1,2], bar_heights)
        subplot_1.set_xticks([1, 2])
        subplot_1.set_xticklabels(('NEG', 'POS'))
        subplot_1.set_title('Pos/Neg Counts')

        # confidence plot
        neg_mean_confidence = 0
        pos_mean_confidence = 0
        if len(neg_confidence_list) > 0:
            neg_mean_confidence = sum(neg_confidence_list) * 1.0 / len(neg_confidence_list)
        if len(pos_confidence_list) > 0:
            pos_mean_confidence = sum(pos_confidence_list) * 1.0 / len(pos_confidence_list)
        bar_heights = [neg_mean_confidence, pos_mean_confidence]
        subplot_2.clear()
        subplot_2.bar([1,2], bar_heights)
        subplot_2.set_xticks([1, 2])
        subplot_2.set_xticklabels(('NEG', 'POS'))
        subplot_2.set_title('Mean Confidence')

        # moving average plot for last 200 values
        lookback_number = min(int(len(all_sentiments_list) / 3), 200)
        average_array = np.ones((lookback_number,)) / lookback_number
        moving_averages = np.convolve(all_sentiments_list, average_array, mode='valid')
        # look at most recent 50000
        if len(moving_averages) > 5000:
            moving_averages = moving_averages[-5000:]
        subplot_3.clear()
        subplot_3.hlines(0.5, xmin = 0, xmax = len(moving_averages))
        subplot_3.plot(moving_averages)
        subplot_3.set_ylim(ymin = 0, ymax = 1)
        subplot_3.set_title('Rolling 200-Tweet Average Sentiment')
        subplot_3.set_ylabel('Positivity (0 - 1)')

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
