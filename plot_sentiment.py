import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
subplot = fig.add_subplot(1,1,1)

def animate(index):
    lines = open('live_results.txt','r').readlines()
    if len(lines) > 0:
        positive_count, negative_count = lines[0].split(',')
        bar_heights = [int(negative_count), int(positive_count)]
        subplot.clear()
        subplot.bar([1,2], bar_heights)
        subplot.set_xticks([1, 2])
        subplot.set_xticklabels(('NEG', 'POS'))

ani = animation.FuncAnimation(fig, animate, interval=5000)
plt.show()