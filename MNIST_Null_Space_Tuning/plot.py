import numpy as np
import matplotlib.pyplot as plt

# plot_these = ['mnist_null_space/results/test_accuracy.txt', \
#               'mnist_standard/results/test_accuracy.txt']

plot_these = ['mnist_standard/results/test_accuracy.txt']

x = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 54000]

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


#plt.xticks(x,x)
ax.set_xlabel('Amount of Training Data', fontsize=16)
ax.set_ylabel('Test Accuracy', fontsize=16)
ax.grid(True, which='both')
ax.minorticks_on()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)

#ax.legend(['Null Space CNN', 'Standard CNN'], fontsize=14)
ax.legend(['Standard CNN'], fontsize=14)
plt.show()