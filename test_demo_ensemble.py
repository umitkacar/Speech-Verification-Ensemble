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

# # Process Time
import time

# CNN Model (Resemblyzer)
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

import torch
# # device config
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wave_path_1 = "./voice_test_data_wav/voice_031_001_001.wav"
wave_path_2 = "./voice_test_data_wav/voice_031_002_001.wav"

y1, sr1 = librosa.load(wave_path_1)
x = librosa.feature.mfcc(y1,sr1)

y2, sr2 = librosa.load(wave_path_2)
y = librosa.feature.mfcc(y2,sr2)

dist_MFCC, cost, acc_cost, path = dtw(x.T, y.T, dist=lambda x, y: norm(x - y, ord=2))

print("MFCC_Dist= " + str(dist_MFCC))

if dist_MFCC < 9000:
    print("MFCC_Verification = " + str(bool(True)))
else:
    print("MFCC_Verification = " + str(bool(False)))

encoder = VoiceEncoder()

fpath1 = Path(wave_path_1)
wav1 = preprocess_wav(fpath1)
embed1 = encoder.embed_utterance(wav1)

start = time.time()

fpath2 = Path(wave_path_2)
wav2 = preprocess_wav(fpath2)
embed2 = encoder.embed_utterance(wav2)

f_dist = lambda x, y: norm(x - y, ord=2)
dist_Resemblyzer = f_dist(embed1,embed2)

print("Resemblyzer_Dist= " + str(dist_Resemblyzer))
if dist_Resemblyzer < 0.80:
    print("Resemblyzer_Verification = " + str(bool(True)))
else:
    print("Resemblyzer_Verification = " + str(bool(False)))

if dist_Resemblyzer < 0.80 or dist_MFCC < 9000:
    print("FUSION_Verification = " + str(bool(True)))
else:
    print("FUSION_Verification = " + str(bool(False)))

# PRECISION & RECALL THRESHOLD OR AND DECSION TREE OPTIMIZATION