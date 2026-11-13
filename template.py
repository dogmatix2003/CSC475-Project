# Template matching for CSC475 Chord Recognition Project

import librosa
import json
import pandas as pd
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import libfmp.c3

outputPath = Path(sys.path[0] + "/Template Predictions")
#set up correct path to Databases folder
path = Path(sys.path[0] + "/Databases")

#create list of paths to all .json files
chord_files = [file.as_posix() for file in path.rglob("*.json") if file.is_file()]

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]

#making chord templates with 12 element vectors and assigning them to notes
def chord_templates():
    
    templates = {}
    chordNames = []

    majorIntervals = [0,4,7]
    minorIntervals = [0,3,7]

    for root in range(12):

        #major chords
        major = np.zeros(12)
        for interval in majorIntervals:
            major[(root + interval) % 12] = 1

        majorName = NOTE_NAMES[root]
        templates[majorName] = major
        chordNames.append(majorName)

        #minor chords
        minor = np.zeros(12)
        for interval in minorIntervals:
            minor[(root + interval) % 12] = 1

        minorName = NOTE_NAMES[root] + "m"
        templates[minorName] = minor
        chordNames.append(minorName)

    #nonchord
    templates["N"] = np.zeros(12)
    chordNames.append("N")

    return templates, chordNames


def template_recognition(x, templates, samplingRate, hopLength, nonChordMask):
    chordNames = list(templates.keys())

    #turning templates into a matrix
    templateMatrix = np.array([templates[name] for name in chordNames])

    predictions = []
    
    for i, frame in enumerate(x):
        
        timestamp = i * hopLength / samplingRate
        
        #handling nonchords
        
        if nonChordMask[i]:
            predictions.append((timestamp,"N"))
            continue

        #comparing frame to all templates
        similarities = cosine_similarity(frame.reshape(1,-1), templateMatrix)[0]
        
        bestMatch = np.argmax(similarities)

        predictions.append((timestamp, chordNames[bestMatch]))
    
    return predictions

templates, chordNames = chord_templates()


#parse through .json files and extract relevant chord info
for file in chord_files:
    #opening each json file and accessing the dict
    current_file = json.load(open(file, "r", encoding="utf-8"))

    #grabbing the song name to properly access the dict
    songName = str(file)
    songName = songName.removeprefix((str(path) + '/').replace("\\", '/'))
    songName = songName.removesuffix('.json')

    outputFile = Path(sys.path[0] + "/Template Predictions" + f"/{songName}_Predictions.txt")

    with open(outputFile, "w") as output:

        count = 0
        hopLength = current_file[songName]["metadata"]["hop_length"]
        samplingRate = current_file[songName]["metadata"]["sampling_rate"]

        x = np.asarray(current_file[songName]["features_X"], dtype=np.float32)

        
        #transposing x
        x = x.T

        nonChordMask = np.linalg.norm(x, axis=0) < 0.01
        
        #normalizing it
        x = libfmp.c3.normalize_feature_sequence(x, '2')
        
        #then transposing it back
        x = x.T
        
        predictions = template_recognition(x, templates, samplingRate, hopLength, nonChordMask)

        #Writing all predictions to output file chord_template_predictions.txt
        for timestamp, chord in predictions:
                output.write(f"{timestamp:.2f}s: {chord}\n")

    
    
output.close()


   
    
