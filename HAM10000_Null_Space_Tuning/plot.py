import numpy as np
import matplotlib.pyplot as plt

# plot_these = ['skinlesion_null_space/results/test_accuracy.txt', \
#               'skinlesion_standard/results/test_accuracy.txt']

plot_these = ['skinlesion_standard/results/test_accuracy.txt']

x =  [1000, 2000, 3000, 4000, 5000, 5500, 6000, 6500, 7000, 7500, 7750, 8000]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for path in plot_these:

    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(float(line))

    data = np.flip(data)
    ax.loglog(x, data)
    #ax.plot(x,data)
    #ax.scatter(x,data)


#plt.xticks(x,x)
ax.set_xlabel('Amount of Training Data', fontsize=24)
ax.set_ylabel('Test Accuracy', fontsize=24)
ax.grid(True, which='both')
ax.minorticks_on()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(24)

# ax.legend(['Null Space Densenet', 'Standard Densenet'], fontsize=24)
ax.legend(['Standard Densenet'], fontsize=24)
plt.show()

