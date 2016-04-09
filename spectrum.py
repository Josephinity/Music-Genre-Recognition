from scipy.io import wavfile
from matplotlib import pyplot as plt
from matplotlib.pyplot import specgram
from pylab import *
import numpy as np
import matplotlib


matplotlib.interactive(True)

# Load the data and calculate the time of each sample
fps, music = wavfile.read('./soft/waltz.wav')
data = music[30 * fps: 40 * fps]
times = np.arange(len(data))/float(fps) + 30

# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))
plt.fill_between(times, data[:,0], data[:,1], color='k') 
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
#plt.savefig('plot.png', dpi=100)
#plt.savfig('/Users/xiaobaby/Desktop/pic.png',dpi = 300)
plt.show()