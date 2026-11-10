# Template matching for CSC475 Chord Recognition Project

import librosa
import json
import pandas as pd
import sys
from pathlib import Path
import numpy as np
import libfmp.c3

#set up correct path to Databases folder
path = Path(sys.path[0] + "/Databases")

#create list of paths to all .json files
chord_files = [file.as_posix() for file in path.rglob("*.json") if file.is_file()]

# Making chord labels, including nonchord label if there are nonchords
def get_chord_labels(ext_minor='m', nonchord=False):
    #Making labels for major and minor chords

    chromaLabels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    majorLabels = chromaLabels
    minorLabels = [s + ext_minor for s in chromaLabels]
    chordLabels = majorLabels + minorLabels

    # Adding nonchord label if there is nonchord
    if nonchord is True:
        chordLabels = chordLabels + ['N']

    return chordLabels

#Making 24 (25 if nonchord) chord templates using cyclical shifting
def chord_templates(nonchord = False):
    #Making basic templates for C-Major and C-Minor triads
    majorTemplate = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).T
    minorTemplate = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]).T 
    numChord = 24

    if nonchord:
        numChord = 25
    chordTemplates = np.ones((12, numChord))

    #Making full templates by shifting base templates for major and minor triads
    for shift in range(12):
        chordTemplates[:, shift] = np.roll(majorTemplate, shift)
        chordTemplates[:, shift+12] = np.roll(minorTemplate, shift)
    print(chordTemplates)
    return chordTemplates

# example template recognition code. WILL ALMOST CERTAINLY BE CHANGED
def template_recognition(X, normSim ='1', nonchord=False):
    chordTemplates = chord_templates(nonchord=nonchord)
    XNorm = libfmp.c3.normalize_feature_sequence(X, norm='2')
    chordTemplatesNorm = libfmp.c3.normalize_feature_sequence(chordTemplates, norm='2')

    chordSim = np.matmul(chordTemplatesNorm.T, XNorm)

    if normSim is not None:
        chordSim = libfmp.c3.normalize_feature_sequence(chordSim, norm=normSim)

    chordMaxIndex = np.argmax(chordSim, axis=0)
    chordMax = np.zeros(chordSim.shape).astype(np.int32)

    for n in range(chordSim.shape[1]):
        chordMax[chordMaxIndex[n], n] = 1

    return chordSim, chordMax


chordTemplates = chord_templates(nonchord=True)
#parse through .json files and extract relevant chord info
for file in chord_files:
    #opening each json file and accessing the dict
    current_file = json.load(open(file, "r", encoding="utf-8"))

    #grabbing the song name to properly access the dict
    songName = str(file)
    songName = songName.removeprefix((str(path) + '/').replace("\\", '/'))
    songName = songName.removesuffix('.json')

    count = 0
    for x in current_file[songName]["data"]:
        #chordSim, chordMax = template_recognition(x, normSim='max')
        #print(chordMax)
        #print(current_file[songName]["data"][count]["start time"])
       
        count += 1
    #print(count)
    
