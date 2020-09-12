import librosa
import librosa.display
import matplotlib.pyplot as plt
from dtw import dtw
import numpy as np
from numpy.linalg import norm

#Loading audio files
y1, sr1 = librosa.load('./umit1.wav') 
y2, sr2 = librosa.load('./umit2.wav') 

#Showing multiple plots using subplot
plt.subplot(2, 2, 1) 
mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values
librosa.display.specshow(mfcc1)

plt.subplot(2, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
librosa.display.specshow(mfcc2)

dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=2))
print("The normalized distance between the two : ",dist)   # 0 for similar audios 

plt.subplot(2, 2, 3)
plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
plt.plot(path[0], path[1], 'w')   #creating plot for DTW
plt.xlim((-0.5, cost.shape[0]-0.5))
plt.ylim((-0.5, cost.shape[1]-0.5))
plt.show()  #To display the plots graphically

