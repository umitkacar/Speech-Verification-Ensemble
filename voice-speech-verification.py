#!/usr/bin/env python
# coding: utf-8

import os
import glob
import sys
from pydub import AudioSegment
from pydub.playback import play

# roc curve and auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# precision-recall curve and f1
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

# Matplotlib
from matplotlib import pyplot as plt

# Numpy
import numpy as np
from numpy.linalg import norm

# Voice
import librosa
import librosa.display
from dtw import dtw

# # Converting Data (.wav)
data_path = ['./voice_test_data/']  # Path where the voice/video are located
data_path_save = ['./voice_test_data_wav/']
extension_list = ('*.mpeg', '*.mp4','*.ogg')

# Convert data from '.mpeg', '.mp4','.ogg' to '.wav'
voice_dir = []
for path in data_path:
    for extension in extension_list:
        _list = sorted(glob.glob(path + extension))
        voice_dir += _list

for i in range(len(voice_dir)):
    wav_filename = data_path_save[0] + os.path.basename(voice_dir[i]).split('.')[0] + '.wav'
    AudioSegment.from_file(voice_dir[i]).export(wav_filename, format='wav')


sound = AudioSegment.from_file(wav_filename, format="wav")
play(sound)

wave_list =sorted(glob.glob(data_path_save[0] + '*.wav'))


# ### ROC function

def calculate_results(predictions, labels):
    
    treshold_max = np.max(predictions)
    treshold_min = np.min(predictions)
    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    step = (treshold_max - treshold_min)/3000
    TPR_full = []
    FPR_full = []
    ROC_ACC_full = []
    #print('*****************************************************************************')
    for treshold in np.arange(treshold_min, treshold_max + step, step):
        
        #print(f'Treshold = {treshold:.4f}')
        idx1 = predictions <= treshold
        idx2 = predictions > treshold

        TP = np.sum(labels[idx1] == 1)
        FN = P - TP             
        TN = np.sum(labels[idx2] == 0)
        FP = N - TN
        #print(f'TP = {TP:.0f}, FN = {FN:.0f}, TN = {TN:.0f}, FP = {FP:.0f}')
        
        # roc curve
        TPR = float(TP/P)
        TPR_full.append(TPR)
        TNR = float(TN/N)
        FPR = 1-TNR
        FPR_full.append(FPR)
        ROC_ACC = (TP + TN)/(P + N)
        ROC_ACC_full.append(ROC_ACC)
        #print(f'TPR = {TPR:.4f}, FPR = {FPR:.4f}, ROC_ACC = {ROC_ACC:.4f}')
        
    return TPR_full, FPR_full, ROC_ACC_full


# # Computer Vision Model (MFCC + DTW)

from tqdm import tqdm
mfcc_total = []
for i in tqdm(range(len(wave_list))):
    y, sr = librosa.load(wave_list[i])
    mfcc = librosa.feature.mfcc(y,sr)   #Computing MFCC values
    mfcc_total.append(mfcc)


mfcc_total_labels = []
mfcc_total_dist =  []
for i in tqdm(range(len(wave_list)-1)):
    
    x = mfcc_total[i]
    lx = int(os.path.basename(wave_list[i]).split('_')[1])
    
    for j in range(i+1,len(wave_list)):
        
        y = mfcc_total[j]
        ly = int(os.path.basename(wave_list[j]).split('_')[1])
        
        dist, cost, acc_cost, path = dtw(x.T, y.T, dist=lambda x, y: norm(x - y, ord=2))
        mfcc_total_dist.append(dist)
        
        if lx == ly:
            label = int(1)
            mfcc_total_labels.append(label)
        else:
            label = int(0)
            mfcc_total_labels.append(label) 


print("mfcc total dist = " + str(len(mfcc_total_dist)))
print("mfcc total labels = " + str(len(mfcc_total_labels)))


mfcc_predictions = np.array([mfcc_total_dist])
mfcc_labels = np.array([mfcc_total_labels])


mfcc_TPR, mfcc_FPR, mfcc_accuracy = calculate_results(mfcc_predictions, mfcc_labels)


print("Accuracy = " + str(np.max(mfcc_accuracy)))


# plt.subplot(1, 2, 1)
# plot the roc curve for the model
plt.plot(mfcc_FPR, mfcc_TPR, marker='.', label='MFCC - ROC (TPR-FPR)')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
# show the legend
plt.legend()
# save and show plot
plt.savefig('MFCC Test Results.png')
plt.show()


# #Showing multiple plots using subplot
# plt.subplot(2, 2, 1) 
# mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values
# librosa.display.specshow(mfcc1)
# plt.subplot(2, 2, 2)
# mfcc2 = librosa.feature.mfcc(y2, sr2)
# librosa.display.specshow(mfcc2)

# plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
# plt.plot(path[0], path[1], 'w')   #creating plot for DTW
# plt.xlim((-0.5, cost.shape[0]-0.5))
# plt.ylim((-0.5, cost.shape[1]-0.5))
# plt.show()  #To display the plots graphically

# # CNN Model (Resemblyzer)

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

# np.set_printoptions(precision=3, suppress=True)
encoder = VoiceEncoder('cpu')
embed_total = []
for i in tqdm(range(len(wave_list))):
    fpath = Path(wave_list[i])
    wav = preprocess_wav(fpath)
    embed = encoder.embed_utterance(wav)
    embed_total.append(embed)

embed_total_labels = []
embed_total_dist =  []
f_dist = lambda x, y: norm(x - y, ord=2)
for i in tqdm(range(len(wave_list)-1)):
    
    x = embed_total[i]
    lx = int(os.path.basename(wave_list[i]).split('_')[1])
    
    for j in range(i+1,len(wave_list)):
        
        y = embed_total[j]
        ly = int(os.path.basename(wave_list[j]).split('_')[1])
        
        dist = f_dist(x,y)
        embed_total_dist.append(dist)
        
        if lx == ly:
            label = int(1)
            embed_total_labels.append(label)
        else:
            label = int(0)
            embed_total_labels.append(label) 


print("embed total dist = " + str(len(embed_total_dist)))
print("embed total labels = " + str(len(embed_total_labels)))


embed_predictions = np.array([embed_total_dist])
embed_labels = np.array([embed_total_labels])


embed_TPR, embed_FPR, embed_accuracy = calculate_results(embed_predictions, embed_labels)

print("Resemblyzer accuracy = " + str(np.max(embed_accuracy)))

# plt.subplot(1, 2, 1)
# plot the roc curve for the model
plt.plot(embed_FPR, embed_TPR, marker='.', label='Resemblyzer ROC (TPR-FPR)')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
# show the legend
plt.legend()
# save and show plot
plt.savefig('Resemblyzer Test Results.png')
plt.show()


# ### Multi-Modal Voice Verification
# (Computer Vision + Deep Learning)

def tanh_normalize(x):
    
    m = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    normalized_x = 0.5 * (np.tanh(0.01 * ((x - m) / std)) + 1)
    
    return normalized_x

embed_normalized = tanh_normalize(embed_predictions[-1])
mfcc_normalized = tanh_normalize(mfcc_predictions[-1])

# Exahustive Search
for i in np.arange(0, 1, 0.1):
    fusion_predictions = i*embed_normalized + (1-i)*mfcc_normalized
    fusion_predictions = np.array([fusion_predictions])
    fusion_labels = embed_labels
    fusion_TPR, fusion_FPR, fusion_accuracy = calculate_results(fusion_predictions, fusion_labels)
    print(f'weight = {i:0.1f} fusion accuracy = {np.max(fusion_accuracy):0.4f}')

fusion_predictions = 0.7*embed_normalized + 0.3*mfcc_normalized
fusion_predictions = np.array([fusion_predictions])
fusion_labels = embed_labels
fusion_TPR, fusion_FPR, fusion_accuracy = calculate_results(fusion_predictions, fusion_labels)
print(f'fusion accuracy = {np.max(fusion_accuracy):0.4f}')

import plotly
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x = embed_FPR, 
    y = embed_TPR,
    name="Resemblyzer"       # this sets its legend entry
))

fig.add_trace(go.Scatter(
    x = mfcc_FPR, 
    y = mfcc_TPR,
    name="MFCC"
))

fig.add_trace(go.Scatter(
    x = fusion_FPR, 
    y = fusion_TPR,
    name="Fusion"
))

fig.update_layout(
    
    title={
        'text': "Multi-Modal Voice Verification",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},

    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    
    font=dict(
        family="Arial, monospace",
        size=20,
        color="#000000"
    )
)

fig.show()

print(f'MFCC accuracy = {np.max(mfcc_accuracy):0.4f}')
print(f'Resemblyzer accuracy = {np.max(embed_accuracy):0.4f}')
print(f'Fusion accuracy = {np.max(fusion_accuracy):0.4f}')


# # Process Time
import time

y1, sr1 = librosa.load(wave_list[1])
x = librosa.feature.mfcc(y1,sr1)

start = time.time()

y2, sr2 = librosa.load(wave_list[2])
y = librosa.feature.mfcc(y2,sr2) 
dist, cost, acc_cost, path = dtw(x.T, y.T, dist=lambda x, y: norm(x - y, ord=2))

print("MFCC Process Time= " + str(time.time() - start))

encoder = VoiceEncoder()
import torch
# # device config
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fpath1 = Path(wave_list[1])
wav1 = preprocess_wav(fpath1)
embed1 = encoder.embed_utterance(wav1)

start = time.time()

fpath2 = Path(wave_list[2])
wav2 = preprocess_wav(fpath2)
embed2 = encoder.embed_utterance(wav2)

dist = f_dist(embed1,embed2)
print("Resemblyzer Process Time= " + str(time.time() - start))


